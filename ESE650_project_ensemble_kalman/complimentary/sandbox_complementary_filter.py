# %% Imports
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from complementary_filter import complementary_filter_update
import pdb
import os

# %%  CSV imu file
start = time.time()
# fname = '../dataset/MachineHall01_reduced/imu0/data.csv'
# fname = '../dataset/MH_02_easy/mav0/imu0/data.csv'
# fname = '../dataset/MH_03_medium/mav0/imu0/data.csv'
# fname = '../dataset/MH_04_difficult/mav0/imu0/data.csv'
fname = os.getcwd() + "/data/euroc_mav_dataset/MH_05_difficult/mav0/imu0/data.csv"

# %%
imu0 = np.genfromtxt(fname, delimiter=',', dtype='float64', skip_header=1)
# gt_data = np.genfromtxt(fname2,delimiter=',', dtype='float64', skip_header=1)

# %% pull out components of data set - different views of matrix

# timestamps in nanoseconds
t = imu0[:, 0]
print(t.shape)

# angular velocities in radians per second
angular_velocity = imu0[:, 1:4]

# linear acceleration in meters per second^2
linear_acceleration = imu0[:, 4:]

# gyro_bias = gt_data[:,11:14]
# acc_bias = gt_data[:,14:]
# pdb.set_trace()
# %% Process the imu data

n = imu0.shape[0]

euler = np.zeros((n, 3))


# acc_bias = np.array([-0.025266, 0.136696, 0.075593])    #mh_01
# gyro_bias = np.array([-0.00317, 0.021267, 0.078502])    #mh_01

# acc_bias = np.array([-0.024346, 0.144439, 0.06754])    #mh_02
# gyro_bias = np.array([-0.002535, 0.021162, 0.07717])    #mh_02

# acc_bias = np.array([-0.022996, 0.125896, 0.057076])    #mh_03
# gyro_bias = np.array([-0.002571, 0.021269, 0.076861])    #mh_03

# acc_bias = np.array([-0.026895, 0.13691, 0.059287])    #mh_04
# gyro_bias = np.array([-0.002133, 0.021059, 0.076659])    #mh_04

acc_bias = np.array([-0.020544, 0.124837, 0.0618])    #mh_05
gyro_bias = np.array([-0.001806, 0.02094, 0.07687])    #mh_05



R = Rotation.identity()
for i in range(1, n):
    # print(i)
    dt = (t[i] - t[i - 1]) * 1e-9
    R = complementary_filter_update(R, angular_velocity[i - 1] - gyro_bias, linear_acceleration[i] - acc_bias, dt)
    euler[i] = R.as_euler('XYZ', degrees=True)

# %% Plots

t2 = (t - t[0]) * 1e-9

end = time.time()
print("runtime: ", (end - start))

temp = np.concatenate((t.reshape((-1,1)), euler), axis = 1)
np.save('complementary_data_mh05_v3', temp)
# pdb.set_trace()

fig = plt.figure()
plt.plot(t2, euler[:, 0], 'b', label='yaw')
plt.plot(t2, euler[:, 1], 'g', label='pitch')
plt.plot(t2, euler[:, 2], 'r', label='roll')
plt.ylabel('degrees')
plt.xlabel('seconds')
plt.title('Attitude of Quad')
plt.legend()
plt.savefig('full_data_result.png')
plt.show()
