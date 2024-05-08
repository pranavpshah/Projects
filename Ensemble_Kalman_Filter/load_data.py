import numpy as np
from scipy.spatial.transform import Rotation
import os

class load_data():
    """
    A class used to load in data that can be used accross any files.
    Handles all rotations needed from VIO world frame to VICON world frame.
    """
    def __init__(self, path_euroc, path_estimate):
        """
        Inputs: 
        1) path_euroc = the path to the parent folder of the Euroc dataset
        2) path_estimate = the path to the folder where the state estimate data is located
        """
        ### load data from the correct directory
        self.data_path = path_estimate
        self.gt_path = path_euroc # the ground truth path

    def load_msckf(self, dataset=1):
        """Get estimate data for the msckf - position and orientation"""
        msckf_data = np.load(os.path.join(self.data_path, "msckf_data" + str(dataset) + ".npy" ), allow_pickle=True)
        msckf_timestamp = msckf_data[:,0] # for the 0th dataset
        msckf_position = np.stack(msckf_data[:,1]) # the np.stack() converts the array to arrays to 2d array to actually use
        msckf_velocity = np.stack(msckf_data[:,2])
        msckf_quat = np.stack(msckf_data[:,3]) # x,y,z,w
        msckf_rpy = Rotation.from_quat(msckf_quat).as_euler('XYZ', degrees=True)
        
        return msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_rpy

    def load_eskf(self, dataset=1):
        """Get estimate data for the eskf - position and orientation"""
        eskf_data = np.load(os.path.join(self.data_path, "eskf_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        eskf_timestamp = eskf_data[:,0] # for the 0th dataset
        eskf_position = np.stack(eskf_data[:,1]).reshape(-1,3) # the np.stack() converts the array to arrays to 2d array to actually use
        eskf_velocity = np.stack(eskf_data[:,2]).reshape(-1,3)
        eskf_quat = np.stack(eskf_data[:,3]) # x,y,z,w

        # Convert the reading from the filter world frame (first frame is the world frame) to the vicon world frame (position)
        if (dataset == 1):
            initial_quat = np.array([-0.153,-0.8273,-0.08215,0.5341]) # x, y, z, w
            initial_pos = np.array([4.688,-1.786,0.783]).transpose()
        elif (dataset == 2):
            initial_quat = np.array([-0.12904, -0.810903, -0.06203, 0.567395]) # x, y, z, w
            initial_pos = np.array([4.62115, -1.837605, 0.739627]) .transpose()
        elif (dataset == 3):
            initial_quat = np.array([-0.156367, -0.776345, -0.072229, 0.606317]) # x, y, z, w
            initial_pos = np.array([4.631117, -1.786812, 0.577113]).transpose()
        elif (dataset == 4):
            initial_quat = np.array([-0.76113, -0.355916, -0.485843, 0.240749])  # x, y, z, w
            initial_pos = np.array([4.677066, -1.74944, 0.568567]).transpose()
        elif (dataset == 5):
            initial_quat = np.array([-0.75761, -0.348629, -0.497711, 0.238261]) # x, y, z, w
            initial_pos = np.array([4.460675, -1.680515, 0.579614]).transpose()
        else:
            raise ValueError('Dataset does not exist')

        # Convert position into the VICON world frame from the VIO world frame
        R_pos = Rotation.from_quat(initial_quat).as_matrix() # this is the first quaternion from the GROUND TRUTH DATA
        for iter in range(eskf_position.shape[0]):
            eskf_position[iter,:] = (R_pos @ eskf_position[iter,:].transpose()) + initial_pos
        
        # Convert the orientation into the VICON world frame from the VIO world frame
        R = Rotation.from_quat(initial_quat) # this is the first quaternion from the GROUND TRUTH DATA
        mat = Rotation.from_quat(eskf_quat)
        mat = R * mat 
        eskf_rpy = mat.as_euler('XYZ', degrees=True)

        return eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_rpy

    def load_complementary(self, dataset = 1):
        """Get estimate data for the complementary filter - orientation only"""
        complementary_data = np.load(os.path.join(self.data_path, "complementary_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        complementary_timestamp = complementary_data[:,0]
        complementary_rpy = np.stack(complementary_data[:,1:])
        complementary_quat = Rotation.from_euler('XYZ', complementary_rpy, degrees=True)

        # Convert the reading from the filter world frame (first frame is the world frame) to the vicon world frame (position)
        if (dataset == 1):
            initial_quat = np.array([-0.153,-0.8273,-0.08215,0.5341]) # x, y, z, w
            initial_pos = np.array([4.688,-1.786,0.783]).transpose()
        elif (dataset == 2):
            initial_quat = np.array([-0.12904, -0.810903, -0.06203, 0.567395]) # x, y, z, w
            initial_pos = np.array([4.62115, -1.837605, 0.739627]) .transpose()
        elif (dataset == 3):
            initial_quat = np.array([-0.156367, -0.776345, -0.072229, 0.606317]) # x, y, z, w
            initial_pos = np.array([4.631117, -1.786812, 0.577113]).transpose()
        elif (dataset == 4):
            initial_quat = np.array([-0.76113, -0.355916, -0.485843, 0.240749])  # x, y, z, w
            initial_pos = np.array([4.677066, -1.74944, 0.568567]).transpose()
        elif (dataset == 5):
            initial_quat = np.array([-0.75761, -0.348629, -0.497711, 0.238261]) # x, y, z, w
            initial_pos = np.array([4.460675, -1.680515, 0.579614]).transpose()
        else:
            raise ValueError('Dataset does not exist')

        R = Rotation.from_quat(initial_quat) # from VIO world frame to vicon world frame
        mat = R * complementary_quat
        complementary_rpy = mat.as_euler('XYZ', degrees=True)
        return complementary_data, complementary_timestamp, complementary_rpy

    def load_ukf(self, dataset=1):
        """Get estimate data for the ukf - orientation only (for now)"""
        ukf_data = np.load(os.path.join(self.data_path, "ukf_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        ukf_roll = ukf_data[0,:]
        ukf_pitch = ukf_data[1,:]
        ukf_yaw = ukf_data[2,:]
        ukf_rpy = np.hstack((ukf_roll.reshape(-1,1), ukf_pitch.reshape(-1,1), ukf_yaw.reshape(-1,1)))  
        ukf_quat = Rotation.from_euler('XYZ', ukf_rpy, degrees=True)
        
        # Convert the reading from the filter world frame (first frame is the world frame) to the vicon world frame (position)
        if (dataset == 1):
            initial_quat = np.array([-0.153,-0.8273,-0.08215,0.5341]) # x, y, z, w
            initial_pos = np.array([4.688,-1.786,0.783]).transpose()
        elif (dataset == 2):
            initial_quat = np.array([-0.12904, -0.810903, -0.06203, 0.567395]) # x, y, z, w
            initial_pos = np.array([4.62115, -1.837605, 0.739627]) .transpose()
        elif (dataset == 3):
            initial_quat = np.array([-0.156367, -0.776345, -0.072229, 0.606317]) # x, y, z, w
            initial_pos = np.array([4.631117, -1.786812, 0.577113]).transpose()
        elif (dataset == 4):
            initial_quat = np.array([-0.76113, -0.355916, -0.485843, 0.240749])  # x, y, z, w
            initial_pos = np.array([4.677066, -1.74944, 0.568567]).transpose()
        elif (dataset == 5):
            initial_quat = np.array([-0.75761, -0.348629, -0.497711, 0.238261]) # x, y, z, w
            initial_pos = np.array([4.460675, -1.680515, 0.579614]).transpose()
        else:
            raise ValueError('Dataset does not exist')

        R = Rotation.from_quat(initial_quat) # from VIO world frame to vicon world frame
        mat = R * ukf_quat
        ukf_rpy = mat.as_euler('XYZ', degrees=True)

        _, ukf_timestamp, _, _, _ = self.load_gt(dataset) # UKF and GT have the same timestamps
        return ukf_data, ukf_timestamp, ukf_rpy

    def load_gt(self, dataset=1):
        """Get ground truth data for the specified dataset"""
        gt_path2 = self.gt_path + "/MH_0" + str(dataset) # specify the directory
        if dataset == 1 or dataset == 2:
            gt_path2 = gt_path2 + "_easy"
        elif dataset == 3:
            gt_path2 = gt_path2 + "_medium"
        else:
            gt_path2 = gt_path2 + "_difficult"

        gt_data = np.loadtxt(os.path.join(gt_path2, "mav0/state_groundtruth_estimate0/data.csv"), delimiter=",")
        gt_timestamp = gt_data[:,0] # for the 0th dataset
        gt_position = np.stack(gt_data[:,1:4]) # the np.stack() converts the array to arrays to 2d array to actually use
        gt_quat = np.stack(gt_data[:,4:8]) # w,x,y,z
        gt_quat = np.roll(gt_quat, -1, axis = 1) 
        gt_rpy = Rotation.from_quat(gt_quat).as_euler('XYZ', degrees=True)
        gt_velocity = np.stack(gt_data[:,8:11])

        return gt_data, gt_timestamp, gt_position, gt_velocity, gt_rpy


if __name__ == "__main__":
    print("This file is only to load data in other files")
    
    
    ### Initlialize
    dataset = 1
    load_stuff = load_data(path_euroc="./data/euroc_mav_dataset", path_estimate="./data/filter_outputs") # initilize the load_data object
    ukf_data, ukf_timestamp, ukf_rpy = load_stuff.load_ukf(dataset)
    gt_data, gt_timestamp, gt_position, gt_velocity, gt_rpy = load_stuff.load_gt(dataset)
    eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_rpy = load_stuff.load_eskf(dataset)

    import matplotlib.pyplot as plt
    

    # Ground Truth with ESKF
    gt_idx_ukf = []
    for i in range(len(ukf_timestamp)):
        gt_idx_ukf.append(np.argmin(np.abs(gt_timestamp - ukf_timestamp[i])))

    plt.figure(7)
    plt.plot(ukf_rpy[:, 0], label="ukf roll estimate")
    plt.plot(ukf_rpy[:, 1], label="ukf pitch estimate")
    plt.plot(ukf_rpy[:, 2], label="ukf yaw estimate")
    plt.plot(gt_rpy[:, 0], label="gt roll", linestyle='dashdot')
    plt.plot(gt_rpy[:, 1], label="gt pitch", linestyle='dashdot')
    plt.plot(gt_rpy[:, 2], label="gt yaw", linestyle='dashdot', color='k')
    plt.xlabel("timestamp")
    plt.ylabel("angle in degrees")
    plt.title("ukf Orientation Estimate")
    plt.legend()

    # Ground Truth with ESKF
    gt_idx_eskf = []
    for i in range(len(eskf_timestamp)):
        gt_idx_eskf.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))

    plt.figure(4)
    plt.plot(eskf_position[:, 0], label="eskf x-pos estimate")
    plt.plot(eskf_position[:, 1], label="eskf y-pos estimate")
    plt.plot(eskf_position[:, 2], label="eskf z-pos estimate")
    plt.plot(gt_position[gt_idx_eskf][:, 0], label="gt x-pos", linestyle='dashdot')
    plt.plot(gt_position[gt_idx_eskf][:, 1], label="gt y-pos", linestyle='dashdot')
    plt.plot(gt_position[gt_idx_eskf][:, 2], label="gt z-pos", linestyle='dashdot', color='k')
    plt.xlabel("timestamp")
    plt.ylabel("position in meters")
    plt.title("ESKF Position Estimate")
    plt.legend()

    plt.figure(6)
    plt.plot(eskf_rpy[:, 0], label="eskf roll estimate")
    plt.plot(eskf_rpy[:, 1], label="eskf pitch estimate")
    plt.plot(eskf_rpy[:, 2], label="eskf yaw estimate")
    plt.plot(gt_rpy[gt_idx_eskf][:, 0], label="gt roll", linestyle='dashdot')
    plt.plot(gt_rpy[gt_idx_eskf][:, 1], label="gt pitch", linestyle='dashdot')
    plt.plot(gt_rpy[gt_idx_eskf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
    plt.xlabel("timestamp")
    plt.ylabel("angle in degrees")
    plt.title("ESKF Orientation Estimate")
    plt.legend()
    plt.show()
