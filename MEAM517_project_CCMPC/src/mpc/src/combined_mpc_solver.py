# import numpy as np
# import matplotlib.pyplot as plt
# from numpy.linalg import inv
# from numpy.linalg import cholesky
# from math import sin, cos
# import math
# from scipy.interpolate import interp1d
# from scipy.integrate import ode
# from scipy.integrate import solve_ivp
# from scipy.linalg import expm
# from scipy.linalg import solve_continuous_are
    
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables


import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist

from scipy.spatial.transform import Rotation as R 

################################ Globals



robot_subscribers = None;
robot_publishers = None
robot_goals = None; #Array with current robot goals
robot_states = None; #Arrays with robot states

robot_num = 2;


################################ ROS functions


class ROS(Node):                             # Node that houses the subscriber and updates global var for position

    def __init__(self, num):
        super().__init__('ros_message handler')
        self.ID = num
        self.goal_subscriber = self.create_subscription(
            Odometry,
            "/robot" + str(num) + "/current_goal",         # The topic to listen to
            self.update_goal,
            10)


        self.state_subscriber = self.create_subscription(
            Odometry,
            "/robot" + str(num) + "/state",         # The topic to listen to
            self.update_state,
            10)                     




    def update_state(self, msg):
        global robot_states
        robot_states[self.ID] = msg.pose.pose;                   # Pose contains position(current_pose.position.x/y/z) and orientation (current_pose.orientation.w/x/y/z)
        #self.get_logger().info("pose updated")

    def update_goal(self, msg):
        global robot_goals
        robot_goals[self.ID] = np.array([msg.x , msg.y, 0]);                   # Pose contains position(current_pose.position.x/y/z) and orientation (current_pose.orientation.w/x/y/z)
        #self.get_logger().info("pose updated")


def update():
    global robot_states
    for i in range(1,robot_num+1):
        rclpy.spin_once(robot_subscribers[i]);


class CommandPublisher(Node):

    def __init__(self, robot = 1):
        self.num = robot
        super().__init__('robot_command_publisher')
        self.publisher_ = self.create_publisher(Twist, "robot" + str(robot) + "/combined_cmd", 10)


####################################


class MPC_system(object):

    def __init__(self, Q, R, Qf):
        self.g = 9.81
        self.m = 1
        self.a = 0.25
        self.I = 0.0625
        self.Q = Q # Shape based on state
        self.R = R
        self.Qf = Qf

        # Input limits
        self.umin = np.array([-0.26, -2.84])
        self.umax = np.array([0.26, 2.84])
        
        # self.umin = np.array([-0.26, -1.0])
        # self.umax = np.array([0.26, 1.0])


        self.nx = 3 # x, y, theta
        self.nu = 2 # v, theta dot(omega)

        # iLQR data


    def continuous_time_full_dynamics(self, x, u):
        # Dynamics for the quadrotor
        # TODO: Pranav add dynamics

        theta = -x[2]
        v = -u[0]
        w = -u[1]

        xdot = np.array([v*np.cos(theta),
                        v*np.sin(theta),
                        w])
        return xdot



    def add_initial_state_constraint(self, prog, x, x_current):
        prog.AddBoundingBoxConstraint(x_current, x_current, x[0])

    def add_input_saturation_constraint(self, prog, x, u, N):
        # Constraint on min and max input

        for ui in u:
            prog.AddBoundingBoxConstraint(self.umin[0], self.umax[0], ui[0])
            prog.AddBoundingBoxConstraint(self.umin[1], self.umax[1], ui[1])

    def add_dynamics_constraint(self, prog, x, u, N, dt):
        # Linearized turtlebot dynamcis
        # A, B = self.discrete_time_linearized_dynamics(xr, ur, dt)
        for i in range(N-1):
            # prog.AddLinearEqualityConstraint(A @ x[i,:] + B @ u[i,:] - x[i+1,:], np.zeros(self.nx))
            f = self.continuous_time_full_dynamics(x[i,:], u[i,:])
            prog.AddConstraint(x[i,0] + f[0]*dt - x[i+1,0], 0, 0)
            prog.AddConstraint(x[i,1] + f[1]*dt - x[i+1,1], 0, 0)
            prog.AddConstraint(x[i,2] + f[2]*dt - x[i+1,2], 0, 0)
        

    def add_collision_constraints(prog,all_x, eps = 0.25):
        for x1 in range(0,len(all_x)):
            for x2 in range(x1+1,len(all_x)):
                for i in range(0,x1.shape[0]):
                    prog.AddLinearConstraint((all_x[x1][i,0] -  all_x[x2][i,0])**2 + (all_x[x1][i,1] -  all_x[x2][i,1])**2 > eps);


    def add_cost(self, prog, xe, u, N):
        for i in range(N-1):
            prog.AddQuadraticCost(xe[i,:] @ self.Q @ xe[i,:].T)
            prog.AddQuadraticCost(u[i,:] @ self.R @ u[i,:].T)
        prog.AddQuadraticCost(xe[i,:] @ self.Qf @ xe[i,:].T)	

    def compute_mpc_feedback(self, x_current, x_r, u_r, T):
        '''
        This function computes the MPC controller input u
        '''

        # Parameters for the QP
        N = 10
        cmd_inputs = [];
        mpc_states = [];
        global robot_goals;
        global robot_states;


        prog = MathematicalProgram()
        for i in range(1, robot_num+1):
            
            # Initialize mathematical program and declare decision variables
            update()
            x_current = robot_states[i]
            
            x = np.zeros((N, self.nx), dtype="object")
            for i in range(N):
                x[i] = prog.NewContinuousVariables(self.nx, "x_" + str(i))
            u = np.zeros((N-1, self.nu), dtype="object")
            for i in range(N-1):
                u[i] = prog.NewContinuousVariables(self.nu, "u_" + str(i))

            # Add constraints and cost
            self.add_initial_state_constraint(prog, x, x_current)
            self.add_input_saturation_constraint(prog, x, u, N)
            self.add_dynamics_constraint(prog, x, u, N, T)
            self.add_cost(prog, x-x_r.reshape((1,self.nx)), u, N)
            all_u += [u]
            all_x += [x]
            # Solve the QP
            # solver = OsqpSolver() 
        self.add_collision_constraints(prog, all_x)
        solver = SnoptSolver()
        result = solver.Solve(prog)

        u_mpc = np.zeros(2) # v and theta_dot

        u_mpc = result.GetSolution(u)[0]

        return u_mpc    

    def compute_lqr_feedback(self, x):
        '''
        Infinite horizon LQR controller
        '''
        A, B = self.continuous_time_linearized_dynamics()
        S = solve_continuous_are(A, B, self.Q, self.R)
        K = -inv(self.R) @ B.T @ S
        u = K @ x
        return u
