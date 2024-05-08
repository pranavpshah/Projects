#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist, Point
from meam517_interfaces.srv import Waypoints

from scipy.spatial.transform import Rotation as R 

import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
from single_robot import Robot
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
        self.subscription                               # prevent unused variable warning

    def update_global_state(self, msg):
        global robot_poses
        robot_poses[self.num] = msg.pose.pose;                   # Pose contains position(current_pose.position.x/y/z) and orientation (current_pose.orientation.w/x/y/z)
        #self.get_logger().info("pose updated")




class CommandPublisher(Node):

    def __init__(self, robot):
        self.num = robot
        super().__init__('robot_command_publisher'+str(robot))
        self.publisher_ = self.create_publisher(Twist, "robot" + str(robot) + "/cmd_vel", 10)

class ClientAsync(Node):

    def __init__(self):
        super().__init__('astar_python_client')
        self.cli = self.create_client(Waypoints, 'find_path')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Waypoints.Request()

    def send_request(self, xs, ys, xg, yg):
        self.req.start.x = xs
        self.req.start.y = ys
        self.req.end.x = xg
        self.req.end.y = yg
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()




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
    
    return False


#This function publishes the command U to the robot cmd_publisher refers to
def publish_robot_command (u, robot_id):
    global robot_cmd_publishers;

    cmd = Twist();
    cmd.linear.x = u[0];
    cmd.angular.z = u[1];
    robot_cmd_publishers[robot_id].publisher_.publish(cmd)



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

def get_all_current_posn(num_robots):
    all_pos = []
    for i in range(num_robots):
        all_pos.append(get_robot_state(i+1))

    return all_pos.copy()

def robot_mpc(robot, robot_id, num_robots):
  # TODO: Nice comment for function
  
    dt = 0.1
    goal_radius = 0.1

    ur = np.zeros(robot.nu)
    x = [waypoints[0]]

    while not update_goal(goal_radius, robot_id):


        current_x = get_robot_state(robot_id) # x, y, theta
        all_pos = get_all_current_posn(num_robots)

        xr = current_goal

        current_u_command = robot.compute_mpc_feedback(current_x, xr, ur, all_pos, dt)
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

        ## check if sleep is required in multi-agent scenario
        # time.sleep(0.1)


    current_u_command = np.zeros(2) # STOP AT GOAL
    publish_robot_command(current_u_command, robot_id)

    x = np.array(x)
    plt.plot(x[:, 0], x[:, 1], "b-", label="MPC path")
    plt.plot(waypoints[:, 0], waypoints[: ,1], "rx", label="Waypoints")
    plt.legend()
    file_path = "graphs/path_"+str(robot_id)
    plt.savefig(file_path+".png")
    np.save(file_path+".npy", x)

def main(args):
    # rclpy.init(args=args)
  
    global waypoints
    global current_goal
    global current_waypoint_index
    global num_robots

    if(len(sys.argv)!=5):
        print ("Please enter number of robots, robot_id, goal_x, goal_y as node arguments")
        return 0


    num_robots = int(sys.argv[1]);
    robot_id = int(sys.argv[2]);

    initialze_ros(num_robots+1) # ROBOTS ARE INDEXED FROM 1
    Q = np.diag([5, 5, 0])
    R = np.diag([0.5, 0.5])
    Qf = Q*2
    bot_radius = 0.15
    epsilon = 0.2

    robot = Robot(Q, R, Qf, bot_radius, epsilon, robot_id);

    minimal_client = ClientAsync()
    start_pos = get_robot_state(robot_id) 
    response = minimal_client.send_request(round(start_pos[0], 0), round(start_pos[1], 0), float(sys.argv[3]), float(sys.argv[4]))
    size = len(response.path)
    waypoints = np.zeros((size,3))
    for i in range(size):
        waypoints[i,:] = np.array([response.path[i].x, response.path[i].y, 0.0])
    print(waypoints)
    file_path = "graphs/waypoints_"+str(robot_id)
    np.save(file_path+".npy", waypoints)

    current_goal = waypoints[0]
    current_waypoint_index = 0

    robot_mpc(robot, robot_id, num_robots)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main(1)
