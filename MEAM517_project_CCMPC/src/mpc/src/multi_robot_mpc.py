#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist, Point

from scipy.spatial.transform import Rotation as R 

import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
from robot import Robot
import matplotlib.pyplot as plt
import time
import sys

################################################################### Global variables ##################################################################

#current_pose =Pose();                                   # Global variable that updates state when required
current_goal = None;
waypoints = [];                                         # Nx3 array [x,y,th]
#current_goal = [];
current_waypoint_index = 0;                             # The index of the current waypoint that we are tracking - index of goal
robot_cmd_publishers = [];
robot_pose_subscribers = [];
robot_poses = [];
num_robots = 0;
#robot_id = 1;
control_mode = "combined"
combined_cmd = None

######################################################################### ROS #########################################################################


class PoseSubscriber(Node):                             # Node that houses the subscriber and updates global var for position

    def __init__(self, robot_number):
        super().__init__('pose_subscriber'+str(robot_number))
        self.num = robot_number
        self.subscription = self.create_subscription(
            Odometry,
            "/robot" + str(robot_number) + "/state",         # The topic to listen to
            self.update_global_state,
            10)


        self.combined_cmd = self.create_subscription(
            Twist,
            "/robot" + str(robot_number) + "/combined_cmd",         # The topic to listen to
            self.update_global_state,
            10)

        self.subscription                               # prevent unused variable warning

    def update_global_state(self, msg):
        global robot_poses
        robot_poses[self.num] = msg.pose.pose;                   # Pose contains position(current_pose.position.x/y/z) and orientation (current_pose.orientation.w/x/y/z)
        #self.get_logger().info("pose updated")

    def update_combined_cmd(self, msg):
        global combined_cmd;
        combined_cmd = msg;
        # Pose contains position(current_pose.position.x/y/z) and orientation (current_pose.orientation.w/x/y/z)
        #self.get_logger().info("pose updated")




class CommandPublisher(Node):

    def __init__(self, robot):
        self.num = robot
        super().__init__('robot_command_publisher'+str(robot))
        self.vel_publisher = self.create_publisher(Twist, "robot" + str(robot) + "/cmd_vel", 10)
        self.goal_publisher = self.create_publisher(Point, "robot" + str(robot) + "/current_goal", 10)




# This function returns the current state of the robot in [x,y,theta] form. Theta is in Radians I think?
# def get_current_state():
#     update()
#     global current_pose;
#     curr = current_pose;
#     x = curr.position.x;
#     y = curr.position.y;
#     r = R.from_quat([ curr.orientation.w, curr.orientation.x, curr.orientation.y, curr.orientation.z])
#     theta = r.as_euler('zyx')
#     return np.array([x,y,theta[2]]);

def get_robot_state(robot_id):

    update_all();
    global robot_poses;
    pose = robot_poses[robot_id];

    x = pose.position.x;
    y = pose.position.y;
    r = R.from_quat([ pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    theta = r.as_euler('zyx')
    
    return np.array([x,y,theta[2]]);






#This function checks if the robot's current state is within the radius of the goal, and if so, it changes the goal to the next one. Returns true if the final goal is reached or false otherwise. 
def update_goal(radius, robot_id):
    global waypoints;
    global current_waypoint_index;
    global current_goal;

    state = get_robot_state(robot_id);


    if( np.sqrt((current_goal[0]-state[0])**2 + (current_goal[1]-state[1])**2 ) < radius):
        current_waypoint_index = current_waypoint_index+1;
        if(current_waypoint_index >= waypoints.shape[0]):
            print("GOAL REACHED");
            return True;

        current_goal = waypoints[current_waypoint_index,:]
    
    publish_robot_goal(current_goal, robot_id);
    
    return False


#This function publishes the command U to the robot cmd_publisher refers to
def publish_robot_command (u, robot_id):
    global robot_cmd_publishers;

    cmd = Twist();
    cmd.linear.x = u[0];
    cmd.angular.z = u[1];
    robot_cmd_publishers[robot_id].vel_publisher.publish(cmd)

def publish_robot_goal (goal, robot_id):
    global robot_cmd_publishers;
    output = Point();
    output.x = goal[0];
    output.y = goal[1];
    #output.z = goal[2];
    robot_cmd_publishers[robot_id].goal_publisher.publish(output)

# Get the cmd_vel sent by the combined MPC solver back to U,V to simulate communication
def get_combined_control_command(robot_id):
    global robot_pose_subscribers;
    global combined_cmd;
    update(robot_id);
    return [combined_cmd.linear.x, combined_cmd.angular.z]



def update(robot_id):
    rclpy.spin_once(robot_pose_subscribers[robot_id]);

def update_all():
    global num_robots
    for i in range(1,num_robots+1):
        update(i);


def initialze_ros(robot_count = 3):
    global robot_pose_subscribers;
    global robot_cmd_publishers;
    global robot_poses;
    global num_robots;


    rclpy.init()
    robot_cmd_publishers = [None];
    robot_pose_subscribers = [None];
    robot_poses = [Pose()];
    
    #num_robots = robot_count

    
    for i in range(1,robot_count):
        robot_poses += [Pose()];
        robot_cmd_publishers += [CommandPublisher(i)];
        robot_pose_subscribers += [PoseSubscriber(i)];




######################################################################### MPC #########################################################################


def robot_mpc(robot, robot_id ):
  # TODO: Nice comment for function
    global control_mode
    dt = 0.15
    goal_radius = 0.1

    ur = np.zeros(robot.nu)
    x = [waypoints[0]]
    while not update_goal(goal_radius, robot_id):


        current_x = get_robot_state(robot_id) # x, y, theta
        xr = current_goal

        if (control_mode == "combined"):
        
            current_u_command = get_combined_control_command(robot_id);
        else:
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
        publish_robot_command(current_u_real,robot_id)
        time.sleep(dt)


    current_u_command = np.zeros(2) # STOP AT GOAL
    publish_robot_command(current_u_command, robot_id)

    x= np.array(x)
    plt.plot(x[:, 0], x[:, 1], "b-", label="MPC path")
    plt.plot(waypoints[:, 0], waypoints[: ,1], "rx", label="Waypoints")
    plt.legend()
    plt.savefig("path.png")

def main(args):

  
    global waypoints
    global current_goal
    global current_waypoint_index
    global num_robots
    global control_mode

    if(len(sys.argv) < 3):
        print ("Please enter number of robots and this node robot id as arguments")
        return;


    num_robots = int(sys.argv[1]);
    robot_id = int(sys.argv[2]);
    control_mode = "notcombined"

    initialze_ros(num_robots+1) # ROBOTS ARE INDEXED FROM 1
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
                            [ 0,  1,  3.14159265],
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

    robot_mpc(robot,robot_id)



if __name__ == '__main__':
    main(1)
