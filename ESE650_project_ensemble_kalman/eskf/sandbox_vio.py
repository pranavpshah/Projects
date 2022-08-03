#%% Imports

import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import yaml
import stereo
from vio import *
from numpy.linalg import norm
import pdb
import time
import os
# %% Import IMU dataset
#  CSV imu file
time_start = time.time()
# dirname = '../dataset/MachineHall01_reduced/imu0/'
# dirname = '../dataset/MH_01_easy/mav0/imu0/'
# dirname = '../dataset/MH_02_easy/mav0/imu0/'
# dirname = '../dataset/MH_03_medium/mav0/imu0/'
# dirname = '../dataset/MH_04_difficult/mav0/imu0/'
dirname = os.getcwd() + "/data/euroc_mav_dataset/MH_05_difficult/mav0/imu0/"


imu0 = np.genfromtxt(dirname + 'data.csv', delimiter=',', dtype='float64', skip_header=1)

# pull out components of data set - different views of matrix

# timestamps in nanoseconds
imu_timestamp = imu0[:, 0]

# angular velocities in radians per second
angular_velocity = imu0[:, 1:4]

# linear acceleration in meters per second^2
linear_acceleration = imu0[:, 4:]

#%% Read IMU calibration data

with open(dirname + 'sensor.yaml', 'r') as file:
    imu_calib_data = yaml.load(file, Loader=yaml.FullLoader)

gyroscope_noise_density = imu_calib_data['gyroscope_noise_density']
gyroscope_random_walk = imu_calib_data['gyroscope_random_walk']
accelerometer_noise_density = imu_calib_data['accelerometer_noise_density']
accelerometer_random_walk = imu_calib_data['accelerometer_random_walk']

# %% Import stereo dataset
# main_data_dir = "../dataset/MachineHall01_reduced/"
# main_data_dir = '../dataset/MH_01_easy/mav0/'
# main_data_dir = '../dataset/MH_02_easy/mav0/'
# main_data_dir = '../dataset/MH_03_medium/mav0/'
# main_data_dir = '../dataset/MH_04_difficult/mav0/'
main_data_dir = dirname[:-5]

dataset = stereo.StereoDataSet(main_data_dir)

# Extract rotation that transforms IMU to left camera frame
R_LB = dataset.stereo_calibration.tr_base_left[0:3, 0:3].T

print("\n\nR_LB: ", R_LB, "\n\n")

# %% Initialize filter

imu_index = 0
stereo_index = 0
# stereo_index = 100

first_image_timestamp = float(dataset.get_timestamp(stereo_index))
print("\n\nfirst_image_timestep: ", first_image_timestamp, "\n\n")
# pdb.set_trace()
while imu_timestamp[imu_index] < first_image_timestamp:
    imu_index += 1
    
print("imu_index: ", imu_index)
print("\n\nIMU timestamp: ", imu_timestamp[imu_index], "\n\n")

# stereo_pair_2 = dataset.process_stereo_pair(0)
stereo_pair_2 = dataset.process_stereo_pair(stereo_index)
stereo_index += 1

next_image_time = float(dataset.get_timestamp(stereo_index))

last_timestamp = first_image_timestamp

##changes made to this variable
nimages = len(dataset.left_images)-1
# nimages = 200 + stereo_index - 1
# pdb.set_trace()

focal_length = dataset.rectified_camera_matrix[0, 0]
print("\n\nfocal length: ", focal_length, "\n\n")

image_measurement_covariance = ((0.5 / focal_length) ** 2) * np.eye(2)
error_threshold = (10 / focal_length)

# R_correction = Rotation.from_quat(np.array([-0.153, -0.8273, -0.08215, 0.5341]))

# Initialize state
p = np.zeros((3, 1))
v = np.zeros((3, 1))
q = Rotation.identity()
a_b = np.zeros((3, 1))
w_b = np.zeros((3, 1))

# q = R_correction*q
# p = p + np.array([[4.688],[-1.786],[0.783]])

g = R_LB @ linear_acceleration[imu_index]
g *= (-9.8 / norm(g))
g = g.reshape(3, 1)
# pdb.set_trace()
nominal_state = p, v, q, a_b, w_b, g

# Initialize error state covariance
error_state_covariance = np.diag([0, 0, 0, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.02, 0.02, 0.02, 0, 0, 0])

# These variables encode last stereo pose
last_R = nominal_state[2].as_matrix()
last_t = nominal_state[0]

trace_covariance = []
pose = []
n_iterations = 0
eskf_data = []

# pdb.set_trace()
# %% Main Loop
while True:
    
    if imu_index >= imu0.shape[0]:
        print("imu break")
        # pdb.set_trace()
        break

    if stereo_index >= nimages:
        print("stereo break")
        # pdb.set_trace()
        break

    trace_covariance.append(error_state_covariance.trace())
    pose.append((nominal_state[2], nominal_state[0], nominal_state[1], nominal_state[3].copy(), nominal_state[4].copy(), nominal_state[5].copy(), np.array([[last_timestamp]])))

    # Extract prevailing a_m and w_m - transform to left camera frame
    w_m = R_LB @ angular_velocity[imu_index - 1, :].reshape(3, 1)
    a_m = R_LB @ linear_acceleration[imu_index - 1, :].reshape(3, 1)

    t = min(imu_timestamp[imu_index], next_image_time)
    dt = (t - last_timestamp) * 1e-9
    last_timestamp = t

    # Apply IMU update
    error_state_covariance = error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                                                     accelerometer_noise_density, gyroscope_noise_density,
                                                     accelerometer_random_walk, gyroscope_random_walk)
    # print(n_iterations)
    nominal_state = nominal_state_update(nominal_state, w_m, a_m, dt)

    # pdb.set_trace()

    if imu_timestamp[imu_index] <= next_image_time:
        # IMU update
        imu_index += 1
    else:
        # Stereo update
        stereo_pair_1 = stereo_pair_2
        stereo_pair_2 = dataset.process_stereo_pair(stereo_index)

        stereo_index += 1
        next_image_time = float(dataset.get_timestamp(stereo_index))

        temporal_match = stereo.TemporalMatch(stereo_pair_1, stereo_pair_2)

        uvd1, uvd2 = temporal_match.get_normalized_matches(dataset.rectified_camera_matrix, dataset.stereo_baseline)

        innovations = np.zeros((2, uvd1.shape[1]))

        for i in range(0, uvd1.shape[1]):
            # Compute Pw
            u1, v1, d1 = uvd1[:, i]

            if d1 > 0:
                P1 = np.array([u1 / d1, v1 / d1, 1 / d1]).reshape(3, 1)

                Pw = last_R @ P1 + last_t

                # Extract uv
                uv = uvd2[0:2, i].reshape(2, 1)

                nominal_state, error_state_covariance, inno = measurement_update_step(nominal_state,
                                                                                      error_state_covariance,
                                                                                      uv, Pw, error_threshold,
                                                                                      image_measurement_covariance)

                innovations[:, i] = inno.ravel()

        count = (norm(innovations, axis=0) < error_threshold).sum()

        pixel_error = np.median(abs(innovations), axis=1) * focal_length

        # print("{} / {} inlier ratio, x_error {:.4f}, y_error {:.4f}, norm_v {:.4f}".format(count, uvd1.shape[1],
        #                                                                                    pixel_error[0],
        #                                                                                    pixel_error[1],
        #                                                                                    norm(nominal_state[1])))

        # These variables encode last stereo pose
        last_R = nominal_state[2].as_matrix()
        last_t = nominal_state[0]

        # pdb.set_trace()
        eskf_data.append([last_timestamp, nominal_state[0], nominal_state[1], nominal_state[2].as_quat()])

    n_iterations += 1

# %% Gather results
print("Number of iterations: ", n_iterations)
n = len(pose)

euler = np.zeros((n, 3))
translation = np.zeros((n, 3))
velocity = np.zeros((n, 3))
a_bias = np.zeros((n, 3))
w_bias = np.zeros((n, 3))
gravity = np.zeros((n, 3))
timestamps = np.zeros((n,1))
data = []
for (i, p) in enumerate(pose):
    euler[i] = p[0].as_euler('XYZ', degrees=True)
    translation[i] = p[1].ravel()
    velocity[i] = p[2].ravel()
    a_bias[i] = p[3].ravel()
    w_bias[i] = p[4].ravel()
    gravity[i] = p[5].ravel()
    timestamps[i] = p[6].ravel()
    # data.append([timestamps[i], translation[i], velocity[i], p[0].as_quat()])

np.save('eskf_data5', eskf_data)

# %% Plot trace of covariance matrix

# pdb.set_trace()
time_end = time.time()
print("runtime: ", (time_end - time_start))

plt.plot(trace_covariance)
plt.title('Trace of covariance matrix')
plt.savefig('covariance_trace.png')
plt.show()

# %% Plot results

fig = plt.figure()

plt.subplot(121)
plt.plot(euler[:, 0], label='yaw')
plt.plot(euler[:, 1], label='pitch')
plt.plot(euler[:, 2], label='roll')
plt.ylabel('degrees')
plt.xlabel('Number of iterations')
plt.title('Attitude of Quad')
plt.legend()

plt.subplot(122)
plt.plot(translation[:, 0], label='Tx')
plt.plot(translation[:, 1], label='Ty')
plt.plot(translation[:, 2], label='Tz')
plt.ylabel('meters')
plt.xlabel('Number of iterations')
plt.title('Position of Quad')
plt.legend()

plt.savefig('pose_transaltion.png')
plt.show()

# pdb.set_trace()

#%%

plt.figure()
plt.plot(velocity[:, 0], label='vx')
plt.plot(velocity[:, 1], label='vy')
plt.plot(velocity[:, 2], label='vz')
plt.ylabel('meters per second')
plt.xlabel('Number of iterations')
plt.title('Velocity of Quad')
plt.legend()
plt.savefig('velocity.png')
plt.show()

#%%
plt.figure()
plt.plot(a_bias[:, 0], label='ax')
plt.plot(a_bias[:, 1], label='ay')
plt.plot(a_bias[:, 2], label='az')
plt.ylabel('meters per second squared')
plt.xlabel('Number of iterations')
plt.title('Accelerometer Bias')
plt.legend()
plt.savefig('accelerometer_bias.png')
plt.show()


#%%

plt.figure()
plt.plot(w_bias[:, 0], label='wx')
plt.plot(w_bias[:, 1], label='wy')
plt.plot(w_bias[:, 2], label='wz')
plt.ylabel('radians per second')
plt.xlabel('Number of iterations')
plt.title('Gyroscope Bias')
plt.legend()
plt.savefig('gyroscope_bias.png')
plt.show()



#%%%

plt.figure()
plt.plot(gravity[:, 0], label='gx')
plt.plot(gravity[:, 1], label='gy')
plt.plot(gravity[:, 2], label='gz')
plt.ylabel('meters per second squared')
plt.xlabel('Number of iterations')
plt.title('Gravity estimate')
plt.legend()
plt.savefig('gravity.png')
plt.show()
