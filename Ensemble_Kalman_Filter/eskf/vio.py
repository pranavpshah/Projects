#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import pdb


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    new_p = np.zeros((3, 1))
    new_v = np.zeros((3, 1))
    new_q = Rotation.identity()

    # pdb.set_trace()
    # g_temp = np.array([[0],[0],[-9.81]])
    R = q.as_matrix()
    # print(R)
    new_p = p + v*dt + (1/2)*((R @ (a_m - a_b)) + g)*(dt**2)    #updating the position in nominal state
    new_v = v + ((R @ (a_m - a_b)) + g)*dt                      #updating velocityin nominal state
    
    axis_angle = np.squeeze((w_m - w_b)*dt)
    q_step = Rotation.from_rotvec(axis_angle)       

    new_q = q*q_step            #updating quaternion in nominal state

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE

    I = np.eye(3)
    # del_t_matrix = dt*np.eye(3)
    R = q.as_matrix()
    am_ab = a_m - a_b
    am_ab_skew = np.array([[0, -am_ab[2,0], am_ab[1,0]],
                           [am_ab[2,0], 0, -am_ab[0,0]],
                           [-am_ab[1,0], am_ab[0,0], 0]])
    axis_angle = np.squeeze((w_m - w_b)*dt)
    R_axis_angle = Rotation.from_rotvec(axis_angle).as_matrix()

    #making Fx matrix
    Fx = np.eye(18)

    Fx[:3,3:6] = I*dt
    Fx[3:6,6:9] = -(R @ am_ab_skew)*dt
    Fx[3:6,9:12] = -R*dt
    Fx[3:6,15:] = I*dt
    Fx[6:9,6:9] = R_axis_angle.T
    Fx[6:9,12:15] = -I*dt

    #making Fi matrix
    Fi = np.zeros((18,12))
    Fi[3:6,0:3] = I
    Fi[6:9,3:6] = I
    Fi[9:12,6:9] = I
    Fi[12:15,9:] = I

    #noise covariance matrix Qi
    V_i = (accelerometer_noise_density**2)*(dt**2)*I
    theta_i = (gyroscope_noise_density**2)*(dt**2)*I
    A_i = (accelerometer_random_walk**2)*dt*I
    ohm_i = (gyroscope_random_walk**2)*dt*I

    Qi = np.eye(12)
    Qi[0:3,0:3] = V_i
    Qi[3:6,3:6] = theta_i
    Qi[6:9,6:9] = A_i
    Qi[9:,9:] = ohm_i

    #updating covariance matrix
    new_P = (Fx @ error_state_covariance @ (Fx.T)) + (Fi @ Qi @ (Fi.T))

    return new_P


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    innovation = np.zeros((2, 1))
    I = np.eye(18)
    R = q.as_matrix()
    Pc = R.T @ (Pw - p)
    #normalized Pc coordinates
    normalized_Pc = np.array([[Pc[0,0]/Pc[2,0]], [Pc[1,0]/Pc[2,0]]])
    # Pc_norm = np.array([[Pc[0,0]/Pc[2,0]], [Pc[1,0]/Pc[2,0]], [1]])
    innovation = uv - normalized_Pc
    dz_dPc = (1/Pc[2,0])*np.array([[1, 0, -normalized_Pc[0,0]],
                                   [0, 1, -normalized_Pc[1,0]]])

    #check for inlier
    if(np.linalg.norm(innovation) < error_threshold):
        H = np.zeros((2, 18))
        hat_Pc = np.array([[0, -Pc[2,0], Pc[1,0]],
                           [Pc[2,0], 0, -Pc[0,0]],
                           [-Pc[1,0], Pc[0,0], 0]])
        H[:,6:9] = dz_dPc @ hat_Pc
        H[:,0:3] = dz_dPc @ (-R.T)

        #computing kalman gain
        K = error_state_covariance @ H.T @ np.linalg.inv((H @ error_state_covariance @ H.T) + Q)

        #updating error_covariance_matrix
        error_state_covariance = ((I - (K @ H)) @ error_state_covariance @ (I - (K @ H)).T) + (K @ Q @ K.T)

        #find error state
        del_state = K @ innovation

        del_theta = del_state[6:9,0]
        del_q = Rotation.from_rotvec(del_theta)

        #propagating the nominal states
        new_q = q*del_q
        q = new_q

        p += del_state[0:3,0].reshape((-1,1))
        v += del_state[3:6,0].reshape((-1,1))
        a_b += del_state[9:12,0].reshape((-1,1))
        w_b += del_state[12:15,0].reshape((-1,1))
        g += del_state[15:,0].reshape((-1,1))

        
    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

# %%
