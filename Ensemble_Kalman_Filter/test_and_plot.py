# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 
import click
import torch

from train import *
from load_data import load_data
import os

# Get user input if they want
@click.command()
@click.option('--dataset', default=1, help='specify the machine hall dataset number. Valid datasets in range [1,5]. Default is 1', type=int)
@click.option('--ensemble', default=True, help='specify which model to use as a boolean: simple average (False) or perceptron/NN (True). Default is True', type=bool)

def main(dataset, ensemble):
    # Run python main.py --help to see how to provide command line arguments
    # Check if the user input makes sense
    if not dataset in [1, 2, 3, 4, 5]:
        raise ValueError('Unknown argument --data %s'%dataset)
    if not ensemble in [True, False]:
        raise ValueError('Unknown argument --ensemble %s'%ensemble)

    ### Define parameters
    # dataset = 1
    match_timesteps = True # if you want the plots to only display pts where the timestamps match. Set to False if you want to debug individual filters
    # perceptron = True # if you want to combine the outputs with a perceptron. Otherwise shows a simple average comparison

    ### Initlialize
    load_stuff = load_data(path_euroc=os.getcwd() + "/data/euroc_mav_dataset", path_estimate=os.getcwd() + "/data/filter_outputs") # initilize the load_data object

    ### Get the data we need
    ukf_data, ukf_timestamp, ukf_rpy = load_stuff.load_ukf(dataset)
    gt_data, gt_timestamp, gt_position, gt_velocity, gt_rpy = load_stuff.load_gt(dataset)
    eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_rpy = load_stuff.load_eskf(dataset)
    msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_rpy = load_stuff.load_msckf(dataset)
    complementary_data, complementary_timestamp, complementary_rpy = load_stuff.load_complementary(dataset)
    
    ## Perceptron Code -----------------------------------------------------------------
    if (ensemble == True):
    
        ## ROLL ----------------------------------------------------------------
        # Match the timesteps of the ESKF with gt in order to make a perceptron of the positions
        match_idx = []
        for i in range(len(eskf_timestamp)):
            match_idx.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))
        # Match the timesteps of the MSCKF with gt in order to evaluate baseline
        match_idx_msckf = []
        for i in range(len(msckf_timestamp)):
            match_idx_msckf.append(np.argmin(np.abs(gt_timestamp - msckf_timestamp[i])))
        # Make a numpy array of all of the filters roll
        x_or_array = np.vstack((eskf_rpy[:, 0], complementary_rpy[match_idx][:, 0], ukf_rpy[match_idx][:, 0], gt_rpy[match_idx][:, 0])).transpose()
       
        # load in the trained model AFTER running perceptron.py
        x_model = x_net()
        x_model.load_state_dict(torch.load(os.getcwd() + '/data/trained_models/x_model.pt'))
        x_model.eval() # dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

        x_test, x_labels_test = x_or_array[:, :-1], x_or_array[:, -1:]
        x_test = torch.from_numpy(x_test) # convert the numpy array to a tensor
        x_labels_test = torch.from_numpy(x_labels_test) # convert the numpy array to a tensor
        test_tds = torch.utils.data.TensorDataset(x_test, x_labels_test)
        x_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)
        criterion = torch.nn.MSELoss()
        x_ls = []
        x_pred = []
        with torch.no_grad():
            for itr, (image, label) in enumerate(x_testloader):
                x_predicted = x_model(image.float())
                loss = criterion(x_predicted, label.float())
                x_ls.append(label.item())
                x_pred.append(x_predicted.item())
            print(f'Loss of test is {loss:.4f}')      

        plt.figure(1)
        plt.rcParams['font.size'] = '30'
        plt.plot(eskf_rpy[:, 0], label="eskf roll", linestyle='dashdot', color='g', linewidth=0.8, alpha=0.6)
        plt.plot(gt_rpy[match_idx][:, 0], label="gt roll", linestyle='solid', color='k')
        plt.plot(ukf_rpy[match_idx][:, 0], label="ukf roll", linestyle='dashdot', color='m', linewidth=0.8, alpha=0.6)
        plt.plot(complementary_rpy[match_idx][:, 0], label="complementary roll estimate", linestyle='dashdot', color='r', linewidth=0.8, alpha=0.6)
        plt.plot(x_pred, label="network roll estimate", linestyle='solid', color='b')
        plt.xlabel("timestamp")
        plt.ylabel("roll estimate in degrees")
        plt.title("Ensemble Filter Estimates for Roll - Network Output", pad=20)
        plt.legend(loc='upper right', prop={'size': 10})

        ukf_loss_roll = np.sqrt(np.mean((gt_rpy[match_idx][:, 0] - ukf_rpy[match_idx][:, 0])**2))
        eskf_loss_roll = np.sqrt(np.mean((gt_rpy[match_idx][:, 0] - eskf_rpy[:, 0])**2))
        complementary_loss_roll = np.sqrt(np.mean((gt_rpy[match_idx][:, 0] - complementary_rpy[match_idx][:, 0])**2))
        new_loss_roll = np.sqrt(np.mean((gt_rpy[match_idx][:, 0] - x_pred)**2))
        msckf_loss_roll = np.sqrt(np.mean((gt_rpy[match_idx_msckf][:, 0] - msckf_rpy[:,0])**2))
        print(f"ukf roll loss: {ukf_loss_roll}, eskf roll loss: {eskf_loss_roll}, complimentary roll loss: {complementary_loss_roll}, model output roll loss: {new_loss_roll}, msckf roll loss: {msckf_loss_roll}")


        ## PITCH ----------------------------------------------------------------
        # Make a numpy array of all of the filters pitch
        y_or_array = np.vstack((eskf_rpy[:, 1], complementary_rpy[match_idx][:, 1], ukf_rpy[match_idx][:, 1], gt_rpy[match_idx][:, 1])).transpose()

        # load in the trained model AFTER running perceptron.py
        y_model = y_net()
        y_model.load_state_dict(torch.load(os.getcwd() + '/data/trained_models/y_model.pt'))
        y_model.eval() # dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

        y_test, y_labels_test = y_or_array[:, :-1], y_or_array[:, -1:]
        y_test = torch.from_numpy(y_test) # convert the numpy array to a tensor
        y_labels_test = torch.from_numpy(y_labels_test) # convert the numpy array to a tensor
        test_tds = torch.utils.data.TensorDataset(y_test, y_labels_test)
        y_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)
        criterion = torch.nn.MSELoss()
        y_ls = []
        y_pred = []
        with torch.no_grad():
            for itr, (image, label) in enumerate(y_testloader):
                y_predicted = y_model(image.float())
                loss = criterion(y_predicted, label.float())
                y_ls.append(label.item())
                y_pred.append(y_predicted.item())
            print(f'MSE loss of test is {loss:.4f}')

        plt.figure(2)
        plt.rcParams['font.size'] = '30'
        plt.plot(eskf_rpy[:, 1], label="eskf pitch", linestyle='solid', color='g', linewidth=0.8, alpha=0.6)
        plt.plot(gt_rpy[match_idx][:, 1], label="gt pitch", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[match_idx][:, 1], label="ukf pitch", linestyle='dashdot', color='m', linewidth=0.8, alpha=0.6)
        plt.plot(complementary_rpy[match_idx][:, 1], label="complementary pitch estimate", linestyle='solid', color='r', linewidth=0.8, alpha=0.6)
        plt.plot(y_pred, label="network pitch estimate", linestyle='dashed', color='b')
        plt.xlabel("timestamp")
        plt.ylabel("pitch estimate in degrees")
        plt.title("Ensemble Filter Estimates for Pitch - Network Output", pad=20)
        plt.legend(loc='upper right', prop={'size': 10})
        

        ukf_loss_pitch = np.sqrt(np.mean((gt_rpy[match_idx][:, 1] - ukf_rpy[match_idx][:, 1])**2))
        eskf_loss_pitch = np.sqrt(np.mean((gt_rpy[match_idx][:, 1] - eskf_rpy[:, 1])**2))
        complementary_loss_pitch = np.sqrt(np.mean((gt_rpy[match_idx][:, 1] - complementary_rpy[match_idx][:, 1])**2))
        new_loss_pitch = np.sqrt(np.mean((gt_rpy[match_idx][:, 1] - y_pred)**2))
        msckf_loss_pitch = np.sqrt(np.mean((gt_rpy[match_idx_msckf][:, 1] - msckf_rpy[:,1])**2))
        print(f"ukf pitch loss: {ukf_loss_pitch}, eskf pitch loss: {eskf_loss_pitch}, complimentary pitch loss: {complementary_loss_pitch}, model output pitch loss: {new_loss_pitch}, msckf pitch loss: {msckf_loss_pitch}")

        ## YAW ----------------------------------------------------------------
        # Make a numpy array of all of the filters yaw
        z_or_array = np.vstack((eskf_rpy[:, 2], complementary_rpy[match_idx][:, 2], ukf_rpy[match_idx][:, 2], gt_rpy[match_idx][:, 2])).transpose()

        # load in the trained model AFTER running perceptron.py
        z_model = z_net()
        z_model.load_state_dict(torch.load(os.getcwd() + '/data/trained_models/z_model.pt'))
        z_model.eval() # dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

        z_test, z_labels_test = z_or_array[:, :-1], z_or_array[:, -1:]
        z_test = torch.from_numpy(z_test) # convert the numpy array to a tensor
        z_labels_test = torch.from_numpy(z_labels_test) # convert the numpy array to a tensor
        test_tds = torch.utils.data.TensorDataset(z_test, z_labels_test)
        z_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)
        criterion = torch.nn.MSELoss()
        z_ls = []
        z_pred = []
        with torch.no_grad():
            for itr, (image, label) in enumerate(z_testloader):
                z_predicted = z_model(image.float())
                loss = criterion(z_predicted, label.float())
                z_ls.append(label.item())
                z_pred.append(z_predicted.item())
            print(f'MSE loss of test is {loss:.4f}')

        plt.figure(3)
        plt.rcParams['font.size'] = '30'
        plt.plot(eskf_rpy[:, 2], label="eskf yaw", linestyle='solid', color='g', linewidth=0.8, alpha=0.6)
        plt.plot(gt_rpy[match_idx][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[match_idx][:, 2], label="ukf yaw", linestyle='dashdot', color='m', linewidth=0.8, alpha=0.6)
        plt.plot(complementary_rpy[match_idx][:, 2], label="complementary yaw estimate", linestyle='solid', color='r', linewidth=0.8, alpha=0.6)
        plt.plot(z_pred, label="network yaw estimate", linestyle='dashed', color='b')
        plt.xlabel("timestamp")
        plt.ylabel("yaw estimate in degrees")
        plt.title("Ensemble Filter Estimates for Yaw - Network Output", pad=20)
        plt.legend(loc='upper right', prop={'size': 10})
        

        ukf_loss_yaw = np.sqrt(np.mean((gt_rpy[match_idx][:, 2] - ukf_rpy[match_idx][:, 2])**2))
        eskf_loss_yaw = np.sqrt(np.mean((gt_rpy[match_idx][:, 2] - eskf_rpy[:, 2])**2))
        complementary_loss_yaw = np.sqrt(np.mean((gt_rpy[match_idx][:, 2] - complementary_rpy[match_idx][:, 2])**2))
        new_loss_yaw = np.sqrt(np.mean((gt_rpy[match_idx][:, 2] - z_pred)**2))
        msckf_loss_yaw = np.sqrt(np.mean((gt_rpy[match_idx_msckf][:, 2] - msckf_rpy[:,2])**2))
        print(f"ukf yaw loss: {ukf_loss_yaw}, eskf yaw loss: {eskf_loss_yaw}, complimentary yaw loss: {complementary_loss_yaw}, model output yaw loss: {new_loss_yaw}, msckf yaw loss: {msckf_loss_yaw}")

        # Include the simple average to compare the ensemble with
        ## Simple Averaging - All Filters
        avg_rpy = np.zeros_like(eskf_rpy)
        avg_rpy[:,0] = ((eskf_rpy[:, 0] + ukf_rpy[match_idx][:, 0] + complementary_rpy[match_idx][:, 0])/3)
        avg_rpy[:,1] = ((eskf_rpy[:, 1] + ukf_rpy[match_idx][:, 1] + complementary_rpy[match_idx][:, 1])/3)
        avg_rpy[:,2] = ((eskf_rpy[:, 2] + ukf_rpy[match_idx][:, 2] + complementary_rpy[match_idx][:, 2])/3)

        avg_loss_roll = np.sqrt(np.mean((gt_rpy[match_idx][:, 0] - avg_rpy[:, 0])**2))
        avg_loss_pitch = np.sqrt(np.mean((gt_rpy[match_idx][:, 1] - avg_rpy[:, 1])**2))
        avg_loss_yaw = np.sqrt(np.mean((gt_rpy[match_idx][:, 2] - avg_rpy[:, 2])**2))
        print(f"avg roll loss: {avg_loss_roll}, avg pitch loss: {avg_loss_pitch}, avg yaw loss: {avg_loss_yaw}")

        plt.figure(4)
        plt.rcParams['font.size'] = '30'
        plt.plot(avg_rpy[:, 0], label="average roll estimate", linestyle='solid', color='r')
        plt.plot(gt_rpy[match_idx][:, 0], label="gt roll", linestyle='dashed', color='k')
        plt.plot(x_pred, label="network roll estimate", linestyle='solid', color='b')
        plt.xlabel("timestamp")
        plt.ylabel("roll estimate in degrees")
        plt.title("Comparison of Simple Avg, Dense NN, and GT - Roll", pad=20)
        plt.legend()

        plt.figure(5)
        plt.rcParams['font.size'] = '30'
        plt.plot(avg_rpy[:, 1], label="average pitch estimate", linestyle='solid', color='r')
        plt.plot(gt_rpy[match_idx][:, 1], label="gt pitch", linestyle='dashed', color='k')
        plt.plot(y_pred, label="network pitch estimate", linestyle='solid', color='b')
        plt.xlabel("timestamp")
        plt.ylabel("pitch estimate in degrees")
        plt.title("Comparison of Simple Avg, Dense NN, and GT - Pitch", pad=20)
        plt.legend()
       
        plt.figure(6)
        plt.rcParams['font.size'] = '30'
        plt.plot(avg_rpy[:, 2], label="average yaw estimate", linestyle='solid', color='r')
        plt.plot(gt_rpy[match_idx][:, 2], label="gt yaw", linestyle='dashed', color='k')
        plt.plot(z_pred, label="network yaw estimate", linestyle='solid', color='b')
        plt.xlabel("timestamp")
        plt.ylabel("yaw estimate in degrees")
        plt.title("Comparison of Simple Avg, Dense NN, and GT - Yaw", pad=20)
        plt.legend()
        
        plt.show()

    elif (ensemble == False):
        """
        This is to compare filters against the ground truth and also to see the simply average comparison
        """
        
        ## Match timesteps for plotting
        # Ground truth with MSCKF
        gt_idx_msckf = []
        for i in range(len(msckf_timestamp)):
            gt_idx_msckf.append(np.argmin(np.abs(gt_timestamp - msckf_timestamp[i])))
        # Ground Truth with ESKF
        gt_idx_eskf = []
        for i in range(len(eskf_timestamp)):
            gt_idx_eskf.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))
        # ESKF with MSCKF
        # match_idx = []
        # for i in range(len(msckf_timestamp)):
        #     match_idx.append(np.argmin(np.abs(eskf_timestamp - msckf_timestamp[i])))
        # Complementary with MSCKF ( this is orientation only)
        # match_idx2 = []
        # for i in range(len(msckf_timestamp)):
        #     match_idx2.append(np.argmin(np.abs(complementary_timestamp - msckf_timestamp[i])))

        plt.figure(1)
        plt.plot(msckf_rpy[:, 0], label="msckf roll estimate")
        plt.plot(msckf_rpy[:, 1], label="msckf pitch estimate")
        plt.plot(msckf_rpy[:, 2], label="msckf yaw estimate")
        plt.plot(gt_rpy[gt_idx_msckf][:, 0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_msckf][:, 1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_msckf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("MSCKF Orientation Estimate")
        plt.legend()

        plt.figure(2)
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

        plt.figure(3)
        plt.plot(ukf_rpy[:, 0], label="ukf roll estimate")
        plt.plot(ukf_rpy[:, 1], label="ukf pitch estimate")
        plt.plot(ukf_rpy[:, 2], label="ukf yaw estimate")
        plt.plot(gt_rpy[:, 0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[:, 1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("UKF Orientation Estimate")
        plt.legend()

        plt.figure(4)
        plt.plot(complementary_rpy[:, 0], label="complementary roll estimate")
        plt.plot(complementary_rpy[:, 1], label="complementary pitch estimate")
        plt.plot(complementary_rpy[:, 2], label="complementary yaw estimate")
        plt.plot(gt_rpy[:, 0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[:, 1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("Complementary Orientation Estimate")
        plt.legend()

        ## Simple Averaging - All Filters
        avg_rpy = np.zeros_like(eskf_rpy)
        avg_rpy[:,0] = ((eskf_rpy[:, 0] + ukf_rpy[gt_idx_eskf][:, 0] + complementary_rpy[gt_idx_eskf][:, 0])/3)
        avg_rpy[:,1] = ((eskf_rpy[:, 1] + ukf_rpy[gt_idx_eskf][:, 1] + complementary_rpy[gt_idx_eskf][:, 1])/3)
        avg_rpy[:,2] = ((eskf_rpy[:, 2] + ukf_rpy[gt_idx_eskf][:, 2] + complementary_rpy[gt_idx_eskf][:, 2])/3)

        avg_loss_roll = np.sqrt(np.mean((gt_rpy[gt_idx_eskf][:, 0] - avg_rpy[:, 0])**2))
        avg_loss_pitch = np.sqrt(np.mean((gt_rpy[gt_idx_eskf][:, 1] - avg_rpy[:, 1])**2))
        avg_loss_yaw = np.sqrt(np.mean((gt_rpy[gt_idx_eskf][:, 2] - avg_rpy[:, 2])**2))
        print(f"avg roll loss: {avg_loss_roll}, avg pitch loss: {avg_loss_pitch}, avg yaw loss: {avg_loss_yaw}")

        plt.figure(5)
        plt.plot(avg_rpy[:, 0], label="average roll estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 0], label="eskf roll estimate", linestyle='solid', color='g')
        plt.plot(gt_rpy[gt_idx_eskf][:, 0], label="gt roll", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[gt_idx_eskf][:, 0], label="ukf roll estimate", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[gt_idx_eskf][:, 0], label="complementary roll estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("roll estimate in degrees")
        plt.title("Ensemble Filter Estimates for Roll - Simple Average")
        plt.legend()

        plt.figure(6)
        plt.plot(avg_rpy[:, 1], label="average pitch estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 1], label="eskf pitch", linestyle='solid', color='g')
        plt.plot(gt_rpy[gt_idx_eskf][:, 1], label="gt pitch", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[gt_idx_eskf][:, 1], label="ukf pitch", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[gt_idx_eskf][:, 1], label="complementary pitch estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("pitch estimate in degrees")
        plt.title("Ensemble Filter Estimates for Pitch - Simple Average")
        plt.legend()
        ukf_loss_pitch = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - ukf_rpy[gt_idx_eskf][:, 1])))
        eskf_loss_pitch = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - eskf_rpy[:, 1])))
        complementary_loss_pitch = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - complementary_rpy[gt_idx_eskf][:, 1])))
        avg_loss_pitch = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - avg_rpy[:, 1])))
        print(f"ukf pitch loss: {ukf_loss_pitch}, eskf pitch loss: {eskf_loss_pitch}, complimentary pitch loss: {complementary_loss_pitch}, model output pitch loss: {avg_loss_pitch}")

        plt.figure(7)
        plt.plot(avg_rpy[:, 2], label="average yaw estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 2], label="eskf yaw", linestyle='solid', color='g')
        plt.plot(gt_rpy[gt_idx_eskf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[gt_idx_eskf][:, 2], label="ukf yaw", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[gt_idx_eskf][:, 2], label="complementary yaw estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("yaw estimate in degrees")
        plt.title("Ensemble Filter Estimates for Yaw - Simple Average")
        plt.legend()
        ukf_loss_yaw = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 2] - ukf_rpy[gt_idx_eskf][:, 2])))
        eskf_loss_yaw = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 2] - eskf_rpy[:, 2])))
        complementary_loss_yaw = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 2] - complementary_rpy[gt_idx_eskf][:, 2])))
        avg_loss_yaw = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 2] - avg_rpy[:, 2])))
        print(f"ukf yaw loss: {ukf_loss_yaw}, eskf yaw loss: {eskf_loss_yaw}, complimentary yaw loss: {complementary_loss_yaw}, model output yaw loss: {avg_loss_yaw}")

        plt.show()

if __name__ == "__main__":    
    print("\nRun python main.py --help to see how to provide command line arguments\n\n")
    main()
    