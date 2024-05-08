import numpy as np
from scipy import io
from quaternion import Quaternion
import matplotlib.pyplot as plt
import tqdm
import os

def estimate_rot():
    """
    Function implementing UFK filter to estimate rotation of moving body

    OUTPUT:
    roll - numpy array containing the mean estimate roll angle for all time steps in dataset
    pitch - numpy array containing the mean estimate pitch angle for all time steps in dataset
    yaw - numpy array containing the mean estimate yaw angle for all time steps in dataset
    """

    dirname = os.getcwd() + '/data/euroc_mav_dataset/MH_05_difficult/mav0/imu0/'
    imu0 = np.genfromtxt(dirname + 'data.csv', delimiter=',', dtype='float64', skip_header=1)
    imu_timestamp = imu0[:, 0]
    gyro = imu0[:, 1:4]
    accel = imu0[:, 4:]
    time_stamp = imu0[:,0].reshape(-1,1)
    T = time_stamp.shape[0]

    # Converting raw IMU data to values.

    # # MH_01_easy
    # accel_values = accel - np.array([-0.025266, 0.136696, 0.075593]).reshape(1,3)
    # gyro_values = gyro - np.array([-0.003172,0.021267,0.078502]).reshape(1,3)
    # accel_values = accel_values / np.linalg.norm(accel_values, axis = 1).reshape(-1,1)

    # # MH_02_easy
    # accel_values = accel - np.array([-0.024346, 0.144439, 0.06754]).reshape(1,3)
    # gyro_values = gyro - np.array([-0.002535,0.021162,0.07717]).reshape(1,3)
    # accel_values = accel_values / np.linalg.norm(accel_values, axis = 1).reshape(-1,1)

    # # MH_03_medium
    # accel_values = accel - np.array([-0.022996,0.125896,0.057076]).reshape(1,3)
    # gyro_values = gyro - np.array([-0.002571,0.021269,0.076861]).reshape(1,3)
    # accel_values = accel_values / np.linalg.norm(accel_values, axis = 1).reshape(-1,1)

    # # MH_04_difficult
    # accel_values = accel - np.array([-0.026895,0.13691,0.059287]).reshape(1,3)
    # gyro_values = gyro - np.array([-0.002133,0.021059,0.076659]).reshape(1,3)
    # accel_values = accel_values / np.linalg.norm(accel_values, axis = 1).reshape(-1,1)

    # MH_05_difficult
    accel_values = accel - np.array([-0.020544,0.124837,0.0618]).reshape(1,3)
    gyro_values = gyro - np.array([-0.001806,0.02094,0.07687]).reshape(1,3)
    accel_values = accel_values / np.linalg.norm(accel_values, axis = 1).reshape(-1,1)
    
    # Mean of the initial state estimation
    x_cap = np.array([1,0,0,0,0,0,0]).reshape(7,1) 
    # Covariance of the initial state estimation
    P = np.eye(6) * 0.005
    # Process noise covariance matrix
    Q = np.eye(6) * 0.005
    # Measurement noise covariance matrix
    R = np.eye(6) * 0.005
    # Initializing the roll, pitch and yaw numpy arrays
    roll = np.zeros((T,))
    pitch = np.zeros((T,))
    yaw = np.zeros((T,))

    wx = np.zeros((T,))
    wy = np.zeros((T,))
    wz = np.zeros((T,))

    # Looping through all entries in dataset
    for iter in tqdm.tqdm(range(T)):
    # for iter in range(T):

        # Creating sigma points
        X = create_sigma_points(P, Q, x_cap)

        # Finding the time difference (dt) between consecutive readings
        if iter == (T-1):
            dt = (time_stamp[iter] - time_stamp[iter - 1]) * 1e-9
        else:
            dt = (time_stamp[iter + 1] - time_stamp[iter]) * 1e-9

        # Transform sigma points
        Y = transform_sigma_points(X, gyro_values[iter,:], dt, x_cap)

        # Finding the mean and covariance of Y matrix
        x_cap_dash, P_dash, W_dash = mean_cov_Y(Y, x_cap)

        # Incorporating the measurement function with transformed sigma points Y
        Z = measurement(Y)

        # Finding the mean and covariance of Z matrix
        z_dash, Pzz = mean_cov_Z(Z)

        # Getting the innovation term
        z_data = np.zeros((6,1))
        z_data[:3,0] = accel_values[iter,:].reshape(3,)
        z_data[3:,0] = gyro_values[iter,:].reshape(3,)
        v = z_data - z_dash

        # Finding Pvv matrix
        Pvv = Pzz + R

        # Finding the kalman gain
        Pxz = (1/12) * (W_dash @ (Z - z_dash).T)
        K = Pxz @ np.linalg.inv(Pvv)

        # Finding the mean and covariance of the new state estimate
        x_cap_next, P_next = mean_cov_next_state(x_cap_dash, K, v, P_dash, Pvv)

        # Updating the previous estimate with new estimate of state mean and covariance
        x_cap = x_cap_next.copy()
        P = P_next.copy()

        # Finding the roll, pitch and yaw of state
        state_quat_obj = Quaternion(scalar = x_cap[0,0], vec = [x_cap[1,0],x_cap[2,0],x_cap[3,0]])
        angle_data = state_quat_obj.euler_angles()
        roll[iter] = (angle_data[0]/np.pi)*180
        pitch[iter] = (angle_data[1]/np.pi)*180
        yaw[iter] = (angle_data[2]/np.pi)*180

    # roll, pitch, yaw are numpy arrays of length T
    return roll, pitch, yaw, time_stamp

def raw_to_value(data, type):
    """
    Function to convert the raw IMU data to values with appropriate units using calibrated bias 
    and sensitivitiy.

    INPUT:
    data - accel or gyro raw data from IMU
    type - type of input data. Can be either "gyro" or "accel"

    OUTPUT:
    data_values - processed sensor values from gyro/accel
    """
    # Initializing array with same shape as input raw data
    data_values = np.zeros_like(data)
    if type == 'accel':
        # Bias and sensitivity for accelerometer IMU in the order x,y,z. Since the x and y axis
        # are flipped, the bias values associated with them will be wrapped around for 16 bit
        # uint datatype
        bias = np.array([-511, -501, 503], dtype = np.uint16)
        bias = bias.astype('float64')
        sensitivity = 332.17

        # The ax and ay component of the accel data is inverted in the IMU, thus we 
        # multiply with -1 
        data_corrected = np.vstack((-data[0],-data[1],data[2])).T
        data_values = ((data_corrected - bias) * 3300/(1023*sensitivity))

    else:
        # Bias and sensitivity for gyroscope IMU in the order x,y,z
        bias = np.array([371.5, 377, 369.5])
        sensitivity = 193.55

        # The order in which the IMU gyro data appears is Wz,Wx,W. Thus we convert
        # to Wx,Wy,Wz order.
        data_corrected = np.vstack((data[1],data[2],data[0])).T
        data_values = ((data_corrected - bias) * 3300/(1023*sensitivity))
    
    return data_values
    
def create_sigma_points(P, Q, x_cap):
    """
    Creating sigma points from previous state estimate, previous state covariance matrix and 
    process noise.

    INPUT:
    P - state covariance matrix from previous state of system
    Q - process noise covariance matrix
    x_cap - state mean estimate values from previous state of system

    OUTPUT:
    X - sigma points
    """
    # Finding the square root using cholesky decomposition
    S = np.linalg.cholesky(P + Q) * np.sqrt(6)
    # Creating the W matrix by coping the -S matrix and stacking with S matrix. Shape of W = (6,12)
    W = np.hstack((S, -S))
    # Initializing the sigma points array. Shape of X = (7,12)
    X = np.zeros((7,12))
    
    # Creating quaternion object for previous state quaternion value 
    q_cap_obj = Quaternion(scalar = x_cap[0,0], vec = x_cap[1:4,0])

    for i in range(12):
        # Creating quaternion object for quaternion vector (3,1) of Wi
        q_w_obj = Quaternion()
        # Converting the quaternion vector (3,1) from Wi to (4,1) quaternion
        q_w_obj.from_axis_angle(W[:3,i])

        # Multipling q_cap and q_w quaternions which is the first 4 entries 
        # of the ith column of X matrix
        q_prod_obj = q_cap_obj * q_w_obj
        q_prod_quat = q_prod_obj.q

        # Finding the last 3 entries of ith column of X matrix 
        X_w = x_cap[4:,:].reshape(3,1) + W[3:,i].reshape(3,1)

        # Combining the first 4 (quaternion) and last 3 (angular velocity) of ith column of X
        X[:4,i] = q_prod_quat.copy().reshape(4,)
        X[4:,i] = X_w.copy().reshape(3,)
    
    return X

def transform_sigma_points(X, gyro, dt, x_cap):
    """
    Function to transform the sigma points (X) using the process function.

    INPUT:
    X - sigma points.
    gyro - processed gyro values for current time 
    dt - time step between current step and next time

    OUTPUT:
    Y - transformed sigma points
    """

    # Initializing the transform points matrix. Shape of Y = (7,12)
    Y = np.zeros((7,12))

    # Creating quaternion for q_delta 
    q_delta_obj = Quaternion()
    q_delta_obj.from_axis_angle(x_cap[4:].reshape(3,)*dt)

    for i in range(12):
        # Creating objects for qx  for multiplication
        qx_obj = Quaternion(scalar = X[0,i], vec = [X[1,i], X[2,i], X[3,i]])

        # Multipling qx and q_delta quaternions for first 4 entries on ith column of Y
        q_prod_obj = qx_obj * q_delta_obj
        q_prod_quat = q_prod_obj.q

        # Combining the first 4 (quaternion) and last 3 (angular velocity) of ith column of Y
        Y[:4,i] = q_prod_quat.copy().reshape(4,)
        Y[4:,i] = (X[4:,i]).reshape(3,)

    return Y

def mean_cov_Y(Y, x_cap):
    """
    Function to find the mean and covariance of transformed sigma points Y

    INPUT:
    Y - transformed sigma points
    x_cap - mean of previous state estimate

    OUTPUT:
    x_cap_dash - mean state estimates after propagate step of UFK
    P_dash - covariance of state after propagate step of UFK
    W_dash - matrix used for Z matrix calculations
    """

    # Threshold for mean convergence
    threshold = 0.001
    # Initial error vector
    e_vec = np.array([1,1,1])
    # Initializing the mean of Y. Shape of x_cap_dash = (7,1)
    x_cap_dash = np.zeros((7,1))

    # Creating quaternion object for previous iteration q_dash quaternion
    qk_obj = Quaternion(scalar = x_cap[0], vec = [x_cap[1], x_cap[2], x_cap[3]])
    
    sum_e_vec = np.zeros((3,12))
    # Running the convergence loop till norm of error vector is close to 0
    while np.linalg.norm(e_vec) > threshold:        
        for i in range(12):
            # Initializing quaternion object for ith column of Y matrix
            qi_obj = Quaternion(scalar = Y[0,i], vec = [Y[1,i], Y[2,i], Y[3,i]])

            # Calculating product of qi and q_k_dash inverse quaternion
            qk_inv_obj = qk_obj.inv()
            q_prod_obj = qi_obj * qk_inv_obj

            # Converting product quaternion to vector form
            q_prod_vec = q_prod_obj.axis_angle().reshape(3,1)

            if np.linalg.norm(q_prod_vec) == 0: 
                sum_e_vec[:,i] = np.zeros(3)
            else:
                sum_e_vec[:,i] = ((-np.pi+np.mod(np.linalg.norm(q_prod_vec)+np.pi, 2*np.pi)) / np.linalg.norm(q_prod_vec)*q_prod_vec).reshape(3,)

        # Finding mean of error vectors
        e_vec = np.mean(sum_e_vec, axis = 1)

        # Creating new q_dash quaternion
        e_vec_obj = Quaternion()
        e_vec_obj.from_axis_angle(e_vec.reshape(3,))
        qk_obj = e_vec_obj * qk_obj

    # Replacing old q_dash with new q_dash value 
    q_dash_next = qk_obj.q

    # Calculating mean state estimates after propagate step of UFK
    x_cap_dash[:4,0] = q_dash_next.reshape(4,)
    x_cap_dash[4:,0] = np.mean(Y[4:,:], axis = 1).reshape(3,)

    # Calculating W_dash with shape (6,12)
    W_dash = np.zeros((6,12))
    for i in range(12):
        W_dash[3:,i] = (Y[4:,i].reshape(3,1) - x_cap_dash[4:,0].reshape(3,1)).reshape(3,)
    W_dash[:3,:] = sum_e_vec.copy()
    
    # Calculating covariance of state after propagate step of UFK
    P_dash = (1/12) * (W_dash @ W_dash.T)

    return x_cap_dash, P_dash, W_dash

def measurement(Y):
    """
    Function to implement the measurement function H() into UFK

    INPUT:
    Y - transformed sigma points

    OUTPUT:
    Z - output of measurement model function on transformed sigma points
    """

    # Initializing the Z matrix. Shape of Z = (6,12)
    Z = np.zeros((6,12))
    # Creating the quaternion object for gravity (g)
    g_obj = Quaternion(scalar = 0, vec = [1,0,0])

    for i in range(12):
        # Creating the quaternion object for quaternion in ith column of Y
        qy_obj = Quaternion(scalar = Y[0,i], vec = [Y[1,i],Y[2,i],Y[3,i]])
        # Getting the inverse quaternion object for the ith column quaternion of Y
        qy_inv_obj = qy_obj.inv()

        # Performing qyi * g * qyi_inv quaternion multiplication
        q_prod2_obj = (qy_inv_obj * g_obj) * qy_obj
        # Getting the 3D vector representation of the product quaternion
        q_prod2_vec = q_prod2_obj.vec()

        # Appending the first 3 elements of ith column of Z with 3D quaternion vector
        Z[:3,i] = q_prod2_vec.reshape(3,)
    # Appending the last 3 elements of ith column of Z with angular velocity values of Y
    Z[3:,:] = Y[4:,:].copy()
    
    return Z

def mean_cov_Z(Z):
    """
    Function to find the mean and covariance of Z matrix

    INPUT:
    Z - output of measurement model function on transformed sigma points

    OUTPUT - 
    z_dash - mean of Z
    Pzz - covariance matrix for Z
    """

    # Finding the mean of Z matrix row wise
    z_dash = np.mean(Z, axis = 1).reshape(6,1)

    # Finding the covariance matrix for Z
    W_z_dash = Z - z_dash
    Pzz = (1/12) * (W_z_dash @ W_z_dash.T)

    return z_dash, Pzz

def mean_cov_next_state(x_cap_dash, K, v, P_dash, Pvv):
    """
    Function to find the update state mean and covariance 

    INPUT:
    x_cap_dash - mean state estimates after propagate step of UFK
    K - kalman gain matrix
    v - innovation term for current time step
    P_dash - covariance of state after propagate step of UFK
    Pvv - covariance of Z matrix coupled with observation model noise

    OUTPUT:
    x_cap_next - updated state mean
    P_next - updated state covariance
    """

    # Finding the gain associated with the innovation term
    gain_innovation = K @ v

    # Finding the quaternion associated with the first 3 terms of the gain_innovation term
    gain_innovation_obj = Quaternion()
    gain_innovation_obj.from_axis_angle(gain_innovation[:3,0].reshape(3,))
    gain_innovation_quat = gain_innovation_obj.q

    # Creating a temporary variable holding the gain_innovation term converted to (7,1) from (6,1)
    temp = np.zeros((7,1))
    temp[:4,0] = gain_innovation_quat.reshape(4,)
    temp[4:,0] = gain_innovation[3:,0].reshape(3,)

    # Quaternion multiplication of converted gain_innovation term and state mean from propagate step
    quat1_obj = Quaternion(scalar = x_cap_dash[0,0], vec = [x_cap_dash[1,0],x_cap_dash[2,0],x_cap_dash[3,0]])
    quat2_obj = Quaternion(scalar = temp[0,0], vec = [temp[1,0],temp[2,0],temp[3,0]])
    q_prod_obj = quat2_obj * quat1_obj
    q_prod_quat = q_prod_obj.q

    # Finding next state estimate
    x_cap_next = np.zeros((7,1))
    x_cap_next[:4,0] = q_prod_quat.reshape(4,)
    x_cap_next[4:,0] = (x_cap_dash[4:,0] + temp[4:,0]).reshape(3,)

    # Finding next state covariance
    P_next = P_dash - (K @ Pvv @ K.T)

    return x_cap_next, P_next


if __name__ == "__main__":

    roll, pitch, yaw, time_stamp = estimate_rot()
    np.save("ukf_data5", np.array([roll,pitch,yaw]))



