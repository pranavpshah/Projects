#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist

from scipy.spatial.transform import Rotation as R 

import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
from robot import Robot
import matplotlib.pyplot as plt
import time

################################################################### Global variables ##################################################################


current_pose =Pose();                                   # Global variable that updates state when required
current_goal = None;
waypoints = [];                                         # Nx3 array [x,y,th]
current_goal = [];
current_waypoint_index = 0;                             # The index of the current waypoint that we are tracking - index of goal
robot_cmd_publisher = 0;
robot_pose_subscriber = 0;

######################################################################### ROS #########################################################################


class PoseSubscriber(Node):                             # Node that houses the subscriber and updates global var for position

    def __init__(self, robot_number = 1):
        super().__init__('pose_subscriber')
        self.num = robot_number
        self.subscription = self.create_subscription(
            Odometry,
            "/robot" + str(robot_number) + "/state",         # The topic to listen to
            self.update_global_state,
            10)
        self.subscription                               # prevent unused variable warning

    def update_global_state(self, msg):
        global current_pose
        current_pose = msg.pose.pose;                   # Pose contains position(current_pose.position.x/y/z) and orientation (current_pose.orientation.w/x/y/z)
        #self.get_logger().info("pose updated")




class CommandPublisher(Node):

    def __init__(self, robot = 1):
        self.num = robot
        super().__init__('robot_command_publisher')
        self.publisher_ = self.create_publisher(Twist, "robot" + str(robot) + "/cmd_vel", 10)




# This function returns the current state of the robot in [x,y,theta] form. Theta is in Radians I think?
def get_current_state():
    update()
    global current_pose;
    curr = current_pose;
    x = curr.position.x;
    y = curr.position.y;
    r = R.from_quat([ curr.orientation.w, curr.orientation.x, curr.orientation.y, curr.orientation.z])
    theta = r.as_euler('zyx')
    return np.array([x,y,theta[2]]);



#This function checks if the robot's current state is within the radius of the goal, and if so, it changes the goal to the next one. Returns true if the final goal is reached or false otherwise. 
def update_goal(radius=0.1):
    global waypoints;
    global current_waypoint_index;
    global current_goal;

    state = get_current_state();


    if( np.sqrt((current_goal[0]-state[0])**2 + (current_goal[1]-state[1])**2 ) < radius):
        current_waypoint_index = current_waypoint_index+1;
        if(current_waypoint_index >= waypoints.shape[0]):
            print("GOAL REACHED");
            return True;

        current_goal = waypoints[current_waypoint_index,:]
    
    return False


#This function publishes the command U to the robot cmd_publisher refers to
def publish_robot_command (u):
    global robot_cmd_publisher;
    cmd = Twist();
    cmd.linear.x = u[0];
    cmd.angular.z = u[1];
    robot_cmd_publisher.publisher_.publish(cmd)



def update():
    rclpy.spin_once(robot_pose_subscriber);


def initialze_ros():
    global robot_pose_subscriber;
    global robot_cmd_publisher;
    global current_pose;

    rclpy.init()

    current_pose = Pose();
    robot_cmd_publisher = CommandPublisher(1);
    robot_pose_subscriber = PoseSubscriber(1);




######################################################################### MPC #########################################################################


def robot_mpc(robot):
  # TODO: Nice comment for function
  
    t0 = 0.0

    dt = 0.15
    goal_radius = 0.1
    ur = np.zeros(robot.nu)
    x = [waypoints[0]]
    while not update_goal(goal_radius):
        current_x = get_current_state() # x, y, theta

        xr = current_goal

        current_u_command = robot.compute_mpc_feedback(current_x, xr, ur, dt)
        #current_u_command = robot.compute_mpc_feedback(0, current_x, ur, dt)
        
        
        print("MPC output: ", current_u_command)
        print("Current goal: ", current_goal)
        print("Current state in loop: ", current_x)


        current_u_real = np.clip(current_u_command, robot.umin, robot.umax)
        # # Autonomous ODE for constant inputs to work with solve_ivp
        # def f(t, x):
        # return robot.continuous_time_full_dynamics(current_x, current_u_real)
        # # Integrate one step
        # sol = solve_ivp(f, (0, dt), current_x, first_step=dt)

        # # Record time, state, and inputs
        # t.append(t[-1] + dt)
        x.append(current_x)
        # u.append(current_u_command)

        # Publish u
        ur = current_u_real
        publish_robot_command(current_u_real)
        time.sleep(dt)


    current_u_command = np.zeros(2) # STOP AT GOAL
    publish_robot_command(current_u_command)

    x= np.array(x)
    plt.plot(x[:, 0], x[:, 1], "b-", label="MPC path")
    plt.plot(waypoints[:, 0], waypoints[: ,1], "rx", label="Waypoints")
    plt.legend()
    plt.savefig("path_noTheta.png")
    plt.savefig("path_noTheta.png")

def main(args):

    initialze_ros()
    global waypoints
    global current_goal
    global current_waypoint_index


    number_of_robots = args
    print("Number of robots: ", number_of_robots)
    Q = np.diag([1.2, 1.2, 0])
    R = np.diag([0.1, 0.15])
    Qf = Q

    robot = Robot(Q, R, Qf);

    # TODO: get map
    
    # Initial state
    start = [(0, 0)]
    goal  = [(5, 5)]

    # TODO: Call astar service
    # waypoints = np.array( [ [ 0.  ,  0.  ,  0.        ],
    #                         [ -1  , 0  , 0],
    #                         [-2.  ,  0.15,  0.1488899583428],
    #                         [-3.  , -0.3 , -0.42285393]])
    waypoints = np.array( [ [ 0,  0,  0.        ],
                            [ 1,  0,  3.14159265],
                            [ 2,  1,  2.35619449],
                            [ 3,  2,  2.35619449],
                            [ 4,  1, -2.35619449],
                            [ 5,  0, -2.35619449],
                            [ 6,  1,  2.35619449],
                            [ 6,  2,  1.57079633],
                            [ 5,  3,  0.78539816],
                            [ 5,  4,  1.57079633],
                            [ 5,  5,  1.57079633]])
    current_goal = waypoints[0]
    current_waypoint_index = 0

    x = [waypoints[0]]
    u = [ur]
    t = 0
    plt.figure()
    for i in range(waypoints.shape[0]-1):
        x_r = waypoints[i+1]

        while (np.linalg.norm(x[-1]-x_r) >= goal_radius) and (t <= tf):
            current_x = x[-1]
            xr = x_r.copy()
            ur = u[-1]
            current_u_command = robot.compute_mpc_feedback(current_x, xr, ur, dt)
            clipped_u = np.clip(current_u_command, umin, umax)

            def f(t, x):
                return robot.continuous_time_full_dynamics(current_x, clipped_u)
            sol = solve_ivp(f, (0, dt), current_x, first_step=dt)

            # xdot = robot.continuous_time_full_dynamics(current_x, clipped_u)
            # new_x = current_x + xdot*dt
            t += dt

            x.append(sol.y[:,-1])
            u.append(clipped_u)
            # print(sol.y)
            print("---------------------------------------")
            print("Goal is: ", xr)
            print("U command: ", clipped_u)
            print("State: ", x[-1])
            print("---------------------------------------")

        if(i == waypoints.shape[0]-1):
            print("Goal reached")

        if(t > tf):
            plot_x = np.array(x)
            plt.plot(plot_x[:,0], plot_x[:,1])
            plt.show()
            break;

    print("All goals reached!!! Hello time: ", t)
    plot_x = np.array(x)
    plt.plot(plot_x[:,0], plot_x[:,1], label='my path')
    plt.plot(waypoints[:,0], waypoints[:,1], 'rx', label='waypoints')
    plt.legend()
    plt.show()


def pseudo_main():
    tf = 10000

    dt = 0.1
    goal_radius = 0.1

    # Q = np.diag([5, 5, 0.01]);
    # R = np.diag([0.1, 0.1]);
    Q = np.diag([5, 5, 0.01]);
    R = np.diag([0.1, 0.2]);
    Qf = Q#np.diag([0.01, 0.01, 0]);
    umin = np.array([-0.26, -1])
    umax = np.array([0.26, 1])

    robot = Robot(Q, R, Qf);
    ur = np.zeros(robot.nu)

    # waypoints = np.array([[0,0,0],
    #                      [0,-1,-1.5708],
    #                      [-2,0.15,0.5218],
    #                      [-3,0.3,0.1489]])
    waypoints = np.array( [ [ 0,  0,  0.        ],
                            [ 1,  0,  3.14159265],
                            [ 2,  1,  2.35619449],
                            [ 3,  2,  2.35619449],
                            [ 4,  1, -2.35619449],
                            [ 5,  0, -2.35619449],
                            [ 6,  1,  2.35619449],
                            [ 6,  2,  1.57079633],
                            [ 5,  3,  0.78539816],
                            [ 5,  4,  1.57079633],
                            [ 5,  5,  1.57079633]])

    x = [waypoints[0]]
    u = [ur]
    t = 0
    plt.figure()
    for i in range(waypoints.shape[0]-1):
        x_r = waypoints[i+1]

        while (np.linalg.norm(x[-1]-x_r) >= goal_radius) and (t <= tf):
            current_x = x[-1]
            xr = x_r.copy()
            ur = u[-1]
            current_u_command = robot.compute_mpc_feedback(current_x, xr, ur, dt)
            clipped_u = np.clip(current_u_command, umin, umax)

            def f(t, x):
                return robot.continuous_time_full_dynamics(current_x, clipped_u)
            sol = solve_ivp(f, (0, dt), current_x, first_step=dt)

            # xdot = robot.continuous_time_full_dynamics(current_x, clipped_u)
            # new_x = current_x + xdot*dt
            t += dt

            x.append(sol.y[:,-1])
            u.append(clipped_u)
            # print(sol.y)
            print("---------------------------------------")
            print("Goal is: ", xr)
            print("U command: ", clipped_u)
            print("State: ", x[-1])
            print("---------------------------------------")

        if(i == waypoints.shape[0]-1):
            print("Goal reached")

        if(t > tf):
            plot_x = np.array(x)
            plt.plot(plot_x[:,0], plot_x[:,1])
            plt.show()
            break;

    print("All goals reached!!! Hello time: ", t)
    plot_x = np.array(x)
    plt.plot(plot_x[:,0], plot_x[:,1], label='my path')
    plt.plot(waypoints[:,0], waypoints[:,1], 'rx', label='waypoints')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    # main(1)
    pseudo_main()
