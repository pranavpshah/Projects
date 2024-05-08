# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import pdb


# %%

def quat_multiply(q1, q2):
    q3 = np.zeros((4,))
    t1 = q1[1:]
    t2 = q2[1:]
    t1 = t1.reshape((3,1))
    t2 = t2.reshape((3,1))
    q3[0] = (q1[0]*q2[0]) - (t1.T @ t2)[0,0]
    q3[1:] = (q1[0]*q2[1:]) + (q2[0]*q1[1:]) + np.cross(q1[1:], q2[1:])

    return q3

def quat_normalize(q):
    norm = np.linalg.norm(q)
    return q/norm

def quat_compliment(q):
    qc = np.zeros((4,))
    qc[0] = q[0]
    qc[1:] = -q[1:].copy()
    return qc

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    # TODO Your code here - replace the return value with one you compute

    q_k_1 = initial_rotation.as_quat()  #(x, y, z, w)
    q_k_1 = np.roll(q_k_1, 1)           #(w, x, y, z)
    
    #new rotation matrix for next frame to prev frame
    axis_angle = angular_velocity*dt
    q_del = Rotation.from_rotvec(axis_angle).as_quat()  #(x, y, z, w)
    q_del = np.roll(q_del, 1)          #(w, x, y, z)
    
    #rotation to 1st frame
    qk = quat_multiply(q_k_1, q_del)
    
    #error magnitude
    em = np.abs((np.linalg.norm(linear_acceleration)) - 9.8)

    #correction gain alpha
    if(em <= 0.98):
        alpha = 1
    elif (em > 0.98 and em < 1.96):
        alpha = 1 - (em - 0.98)/0.96
    else:
        alpha = 0

    
    ak = np.zeros((4,))
    ak[1:] = linear_acceleration
    
    #rotating acceleration vector to 1sr frame
    g_temp = quat_multiply(quat_multiply(qk, ak), quat_compliment(qk))
    
    g_prime = g_temp[1:].copy()
    g_prime = g_prime/np.linalg.norm(g_prime)
    # print(g_prime)
    # exit()
    
    #correction rotation in quaternion form
    q_correction = np.array([np.sqrt((1 + g_prime[0])/2), 0, g_prime[2]/np.sqrt(2*(1 + g_prime[0])), -g_prime[1]/np.sqrt(2*(1 + g_prime[0]))])
    
    #weighted correction
    qI = np.array([1,0,0,0])
    q_prime_correction = (1 - alpha)*qI + alpha*q_correction
    q_prime_correction = quat_normalize(q_prime_correction)
    
    #updated rotation
    q_temp = quat_multiply(q_prime_correction, qk)  #(w, x, y, z)
    q_return = np.roll(q_temp, -1)  #(x, y, z, w)

    return Rotation.from_quat(q_return)
