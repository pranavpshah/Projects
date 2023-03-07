import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE

        ##Implementing the Geometric controller
                
        ##Defining Kp and Kd, diagonal matrices
        Kp = np.array([[22,0,0],[0,22,0],[0,0,15]])
        Kd = np.array([[7.9,0,0],[0,7.9,0],[0,0,6.45]])

        ##desired rddot
        #control equation for position and velocity
        rddot_des = flat_output['x_ddot'] - Kd@(state['v'] - flat_output['x_dot']) - Kp@(state['x'] - flat_output['x'])
        
        F_des = self.mass*rddot_des + np.array([0,0,self.mass*self.g])  #desired force vector
        
        R = Rotation.from_quat(state['q']).as_matrix()  #extracting rotation matrix from the current quaternion

        b3 = (R)@(np.array([[0],[0],[1]]))  #extracting current b3 axis vector from the rotation matrix found above
        
        u1 = ((b3.T)@F_des)[0]  #Finding u1 (Total thrust)
        
        b3_des = F_des/np.linalg.norm(F_des)    #Finding the desired b3 vector that aligns with the desired force vector
        
        a_yaw = np.array([np.cos(flat_output['yaw']), np.sin(flat_output['yaw']), 0])   #finding yaw heading    
        
        b2_des = np.cross(b3_des, a_yaw)    #desired b2 axis orientation
        
        b2_des = b2_des/np.linalg.norm(b2_des)  #normalization

        b1_des = np.cross(b2_des, b3_des)   #desired b1 axis orientation
        
        R_des = np.concatenate((b1_des.reshape((3,1)), b2_des.reshape((3,1)), b3_des.reshape((3,1))), axis = 1) #desired rotation matrix (desired orientation of quadrotor)        
        
        er_temp = 0.5*((R_des.T @ R) - (R.T @ R_des)) #error in quadrotor orientation with respect to current state and desired state
        
        er = np.array([er_temp[2,1], er_temp[0,2], er_temp[1,0]])   #error vetor from skew symmetric matric above
        
        ew = state['w'] - flat_output['yaw_dot']    #motor speed error
    
        ##Defining Kr and Kw, diagonal matrices
        Kr = np.array([[3500,0,0],[0,3500,0],[0,0,403]])   
        Kw = np.array([[107.75,0,0],[0,107.75,0],[0,0,43.75]]) 

        #control equation for orientaion and yaw rate
        u2 = self.inertia @ ((-Kr @ er) - (Kw @ ew))
        
        #matrix to relate [u1 u2] to 4 motor speeds
        mixer = np.array([[self.k_thrust, self.k_thrust, self.k_thrust, self.k_thrust],
                          [0, self.k_thrust*self.arm_length, 0, -self.k_thrust*self.arm_length],
                          [-self.k_thrust*self.arm_length, 0, self.k_thrust*self.arm_length, 0],
                          [self.k_drag, -self.k_drag, self.k_drag, -self.k_drag]])
    
        wrench = np.array([[u1], [u2[0]], [u2[1]], [u2[2]]])
        
        w_temp = np.linalg.inv(mixer) @ wrench
        
        w_temp = np.clip(w_temp, self.rotor_speed_min**2, self.rotor_speed_max**2)  ##clipping motor speeds to min and max rotor speeds given

        w = np.sqrt(w_temp)
        
        cmd_motor_speeds = w.reshape((w.shape[0],))
        
        cmd_thrust = u1.copy()
        
        cmd_moment = u2.copy()
        
        cmd_q = Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
