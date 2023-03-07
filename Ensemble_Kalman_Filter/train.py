## Import the necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tqdm
from load_data import load_data
import click
import os
import pdb

## Define the classes for each of the perceptrons
class x_net(torch.nn.Module):
    """
    A dense neural net for roll
    """
    def __init__(self):
        super(x_net, self).__init__()

        self.model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(True),
        torch.nn.Linear(5, 1),
        )

    def forward(self, x):
        x = self.model(x) # this is the forward pass for the fully connected layer
        return x

class x_tron(torch.nn.Module):
    """
    A single layer perceptron for roll
    """
    def __init__(self):
        super(x_tron, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1) # this is a fully connected layer (single layer perceptron)

    def forward(self, x):
        x = self.fc1(x) # this is the forward pass for the fully connected layer
        return x

class y_net(torch.nn.Module):
    """
    A dense neural net for pitch
    """
    def __init__(self):
        super(y_net, self).__init__()
        
        self.model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(True),
        torch.nn.Linear(5, 1),
        )

    def forward(self, x):
        x = self.model(x) # this is the forward pass for the fully connected layer
        return x

class y_tron(torch.nn.Module):
    """
    A single layer perceptron for pitch
    """
    def __init__(self):
        super(y_tron, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1) # this is a fully connected layer (single layer perceptron)

    def forward(self, x):
        x = self.fc1(x) # this is the forward pass for the fully connected layer
        return x

class z_net(torch.nn.Module):
    """
    A dense neural net for yaw
    """
    def __init__(self):
        super(z_net, self).__init__()
        
        self.model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(True),
        torch.nn.Linear(5, 1),
        )

    def forward(self, x):
        x = self.model(x) # this is the forward pass for the fully connected layer
        return x

class z_tron(torch.nn.Module):
    """
    A single layer perceptron for yaw
    """
    def __init__(self):
        super(z_tron, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1) # this is a fully connected layer (single layer perceptron)

    def forward(self, x):
        x = self.fc1(x) # this is the forward pass for the fully connected layer
        return x

def loss_func(output, target):
    # MSE Loss ##
    diff = (output - target)**2

    # ## Custom Loss ##
    # diff = output - target
    # diff = (diff + 180) % 360 - 180
    
    return torch.mean(torch.abs(diff))

def orientation_perceptron(x_arr, y_arr, z_arr, dense=True):

    epochs = 225
    batch_size = 16

    ## ROLL --------------------------------------------
    print("Training Roll...")
    np.random.shuffle(x_arr)
    x_train, x_test, x_labels_train, x_labels_test = train_test_split(x_arr[:, :-1], x_arr[:, -1:], test_size=0.20, random_state=42)
    x_train = torch.from_numpy(x_train) # convert the numpy array to a tensor
    x_test = torch.from_numpy(x_test) # convert the numpy array to a tensor
    x_labels_train = torch.from_numpy(x_labels_train) # convert the numpy array to a tensor
    x_labels_test = torch.from_numpy(x_labels_test) # convert the numpy array to a tensor
    train_tds = torch.utils.data.TensorDataset(x_train, x_labels_train)
    test_tds = torch.utils.data.TensorDataset(x_test, x_labels_test)
    x_trainloader = torch.utils.data.DataLoader(train_tds, batch_size=batch_size, shuffle=False)
    x_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)

    if (dense == True):
        x_model = x_net()
    else:
        x_model = x_tron()
    optimizer = torch.optim.Adam(x_model.parameters(), lr=1e-2, weight_decay=1e-4) 
    train_mse = []
    # training iterations
    for epoch in tqdm.tqdm(range(epochs)):
        running_loss = 0
        count = 0
        for itr, (image, label) in enumerate(x_trainloader):
            # zero gradient
            optimizer.zero_grad()
            # forward path
            x_predicted = x_model(image.float())
            # target = torch.ones(x_predicted.size(0))
            # loss = criterion(x_predicted, label.float(), target)
            loss = loss_func(x_predicted, label.float())
            # if(itr == 0):
            #     print(f'epoch: {epoch+1}, batch: {itr+1}, loss: {loss.item():.4f}')
            running_loss += loss.item()
            # backpropagating
            loss.backward()
            # optimizes the weights
            optimizer.step()
            count += 1
        train_mse.append(running_loss/(count))
        # if (epoch+1) % 3 == 0:
        #     print(f'epoch: {epoch+1}, loss: {running_loss:.4f}')

    x_ls = []
    x_pred = []
    with torch.no_grad():
        for itr, (image, label) in enumerate(x_testloader):
            x_predicted = x_model(image.float())
            target = torch.ones(x_predicted.size(0))
            # loss = criterion(x_predicted, label.float(), target)
            loss = loss_func(x_predicted, label.float())
            x_ls.append(label.item())
            x_pred.append(x_predicted.item())
        print(f'MSE loss of test is {loss:.4f}')

    torch.save(x_model.state_dict(), os.getcwd() + '/data/trained_models/x_model.pt')

    plt.figure(1)
    plt.plot(train_mse)
    plt.title('Training Loss')

    plt.figure(2)
    plt.plot(x_ls, color = 'g')
    plt.plot(x_pred, color = 'r')
    plt.title('Prediction')
    # plt.show()


    ## PITCH --------------------------------------------
    print("Training Pitch...")
    np.random.shuffle(y_arr)
    y_train, y_test, y_labels_train, y_labels_test = train_test_split(y_arr[:, :-1], y_arr[:, -1:], test_size=0.20, random_state=42)
    y_train = torch.from_numpy(y_train) # convert the numpy array to a tensor
    y_test = torch.from_numpy(y_test) # convert the numpy array to a tensor
    y_labels_train = torch.from_numpy(y_labels_train) # convert the numpy array to a tensor
    y_labels_test = torch.from_numpy(y_labels_test) # convert the numpy array to a tensor
    train_tds = torch.utils.data.TensorDataset(y_train, y_labels_train)
    test_tds = torch.utils.data.TensorDataset(y_test, y_labels_test)
    y_trainloader = torch.utils.data.DataLoader(train_tds, batch_size=batch_size, shuffle=False)
    y_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)

    if (dense == True):
        y_model = y_net()
    else:
        y_model = y_tron()
    optimizer = torch.optim.Adam(y_model.parameters(), lr=1e-2, weight_decay=1e-4) 
    train_mse = []
    # training iterations
    for epoch in tqdm.tqdm(range(epochs)):
        running_loss = 0
        count = 0
        for itr, (image, label) in enumerate(y_trainloader):
            # zero gradient
            optimizer.zero_grad()
            # forward path
            y_predicted = y_model(image.float())
            target = torch.ones(y_predicted.size(0))
            # loss = criterion(y_predicted, label.float(), target)
            loss = loss_func(y_predicted, label.float())
            # if(itr == 0):
            #     print(f'epoch: {epoch+1}, batch: {itr+1}, loss: {loss.item():.4f}')
            running_loss += loss.item()
            # backpropagating
            loss.backward()
            # optimizes the weights
            optimizer.step()
            count += 1
        train_mse.append(running_loss/(count))
        # if (epoch+1) % 3 == 0:
        #     print(f'epoch: {epoch+1}, loss: {running_loss:.4f}')

    y_ls = []
    y_pred = []
    with torch.no_grad():
        for itr, (image, label) in enumerate(y_testloader):
            y_predicted = y_model(image.float())
            target = torch.ones(y_predicted.size(0))
            # loss = criterion(y_predicted, label.float(), target)
            loss = loss_func(y_predicted, label.float())
            y_ls.append(label.item())
            y_pred.append(y_predicted.item())
        print(f'MSE loss of test is {loss:.4f}')

    torch.save(y_model.state_dict(), os.getcwd() + '/data/trained_models/y_model.pt')

    plt.figure(1)
    plt.plot(train_mse)
    plt.title('Training Loss')

    plt.figure(2)
    plt.plot(y_ls, color = 'g')
    plt.plot(y_pred, color = 'r')
    plt.title('Prediction')
    # plt.show()

    ## YAW --------------------------------------------
    print("Training Yaw...")
    np.random.shuffle(z_arr)
    z_train, z_test, z_labels_train, z_labels_test = train_test_split(z_arr[:, :-1], z_arr[:, -1:], test_size=0.20, random_state=42)
    z_train = torch.from_numpy(z_train) # convert the numpy array to a tensor
    z_test = torch.from_numpy(z_test) # convert the numpy array to a tensor
    z_labels_train = torch.from_numpy(z_labels_train) # convert the numpy array to a tensor
    z_labels_test = torch.from_numpy(z_labels_test) # convert the numpy array to a tensor
    train_tds = torch.utils.data.TensorDataset(z_train, z_labels_train)
    test_tds = torch.utils.data.TensorDataset(z_test, z_labels_test)
    z_trainloader = torch.utils.data.DataLoader(train_tds, batch_size=batch_size, shuffle=False)
    z_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)

    if (dense == True):
        z_model = z_net()
    else:
        z_model = z_tron()
    optimizer = torch.optim.Adam(z_model.parameters(), lr=5e-3, weight_decay=1e-4) 
    train_mse = []
    # training iterations
    for epoch in tqdm.tqdm(range(epochs)):
        running_loss = 0
        count = 0
        for itr, (image, label) in enumerate(z_trainloader):
            # zero gradient
            optimizer.zero_grad()
            # forward path
            z_predicted = z_model(image.float())
            target = torch.ones(z_predicted.size(0))
            # loss = criterion(z_predicted, label.float(), target)
            loss = loss_func(z_predicted, label.float())
            # if(itr == 0):
            #     print(f'epoch: {epoch+1}, batch: {itr+1}, loss: {loss.item():.4f}')
            running_loss += loss.item()
            # backpropagating
            loss.backward()
            # optimizes the weights
            optimizer.step()
            count += 1
        train_mse.append(running_loss/(count))
        # if (epoch+1) % 3 == 0:
        #     print(f'epoch: {epoch+1}, loss: {running_loss:.4f}')

    z_ls = []
    z_pred = []
    with torch.no_grad():
        for itr, (image, label) in enumerate(z_testloader):
            z_predicted = z_model(image.float())
            target = torch.ones(z_predicted.size(0))
            # loss = criterion(z_predicted, label.float(), target)
            loss = loss_func(z_predicted, label.float())
            z_ls.append(label.item())
            z_pred.append(z_predicted.item())
        print(f'MSE loss of test is {loss:.4f}')

    torch.save(z_model.state_dict(), os.getcwd() + '/data/trained_models/z_model.pt')

    plt.figure(1)
    plt.plot(train_mse)
    plt.title('Training Loss')

    plt.figure(2)
    plt.plot(z_ls, color = 'g')
    plt.plot(z_pred, color = 'r')
    plt.title('Prediction')
    plt.show()

# Get user input if they want
@click.command()
@click.option('--dataset1', default=1, help='specify the machine hall dataset number to train on. Must be unique from other datasets. Valid datasets in range [1,5]', type=int)
@click.option('--dataset2', default=3, help='specify the machine hall dataset number to train on. Must be unique from other datasets. Valid datasets in range [1,5]', type=int)
@click.option('--dataset3', default=5, help='specify the machine hall dataset number to train on. Must be unique from other datasets. Valid datasets in range [1,5]', type=int)
@click.option('--dense', default=True, help='specify whether to run a dense NN (True), or a perceptron (False). Default set to True.', type=bool)

def main(dataset1, dataset2, dataset3, dense):

    # Run python main.py --help to see how to provide command line arguments
    # Check if the user input makes sense
    if not dataset1 in [1, 2, 3, 4, 5]:
        raise ValueError('Unknown argument --data %s'%dataset1)
    if not dataset2 in [1, 2, 3, 4, 5]:
        raise ValueError('Unknown argument --data %s'%dataset1)
    if not dataset3 in [1, 2, 3, 4, 5]:
        raise ValueError('Unknown argument --data %s'%dataset3)
    if (dataset1 == dataset2 or dataset2 == dataset3 or dataset1 == dataset3):
        raise ValueError('Datasets not unique')
    if (dense not in [True, False]):
        raise ValueError('Please enter a boolean for the dense flag')

    ### Initlialize loading the data object
    load_stuff = load_data(path_euroc=os.getcwd() + "/data/euroc_mav_dataset", path_estimate=os.getcwd() + "/data/filter_outputs") # initilize the load_data object

    ### Get the data we need -- Dataset 1
    ukf_data1, ukf_timestamp1, ukf_rpy1 = load_stuff.load_ukf(dataset1)
    gt_data1, gt_timestamp1, gt_position1, gt_velocity1, gt_rpy1 = load_stuff.load_gt(dataset1)
    eskf_data1, eskf_timestamp1, eskf_position1, eskf_velocity1, eskf_rpy1 = load_stuff.load_eskf(dataset1)
    complementary_data1, complementary_timestamp1, complementary_rpy1 = load_stuff.load_complementary(dataset1)

    # Match the timesteps of the ESKF with gt in order to make a perceptron of the positions
    match_idx1 = []
    for i in range(len(eskf_timestamp1)):
        match_idx1.append(np.argmin(np.abs(gt_timestamp1 - eskf_timestamp1[i])))

    # Make a numpy array of all of the filters x,y,z positions (THIS IS NOW BROKEN)
    # x_pos_array = np.vstack((msckf_position[:, 0], eskf_position[match_idx][:, 0])).transpose()
    # y_pos_array = np.vstack((msckf_position[:, 1], eskf_position[match_idx][:, 1])).transpose()
    # z_pos_array = np.vstack((msckf_position[:, 2], eskf_position[match_idx][:, 2])).transpose()
    # print("TESTING 1: ", x_pos_array.shape, y_pos_array.shape, z_pos_array.shape)
    # Make a numpy array of all of the filters x,y,z positions
    x_or_array1 = np.vstack((eskf_rpy1[:, 0], complementary_rpy1[match_idx1][:, 0], ukf_rpy1[match_idx1][:, 0], gt_rpy1[match_idx1][:, 0]))
    y_or_array1 = np.vstack((eskf_rpy1[:, 1], complementary_rpy1[match_idx1][:, 1], ukf_rpy1[match_idx1][:, 1], gt_rpy1[match_idx1][:, 1]))
    z_or_array1 = np.vstack((eskf_rpy1[:, 2], complementary_rpy1[match_idx1][:, 2], ukf_rpy1[match_idx1][:, 2], gt_rpy1[match_idx1][:, 2]))

    ### Get the data we need -- Dataset 2
    ukf_data2, ukf_timestamp2, ukf_rpy2 = load_stuff.load_ukf(dataset2)
    gt_data2, gt_timestamp2, gt_position2, gt_velocity2, gt_rpy2 = load_stuff.load_gt(dataset2)
    eskf_data2, eskf_timestamp2, eskf_position2, eskf_velocity2, eskf_rpy2 = load_stuff.load_eskf(dataset2)
    complementary_data2, complementary_timestamp2, complementary_rpy2 = load_stuff.load_complementary(dataset2)

    # Match the timesteps of the ESKF with gt in order to make a perceptron of the positions
    match_idx2 = []
    for i in range(len(eskf_timestamp2)):
        match_idx2.append(np.argmin(np.abs(gt_timestamp2 - eskf_timestamp2[i])))
    
    # Make a numpy array of all of the filters x,y,z positions
    x_or_array2 = np.vstack((eskf_rpy2[:, 0], complementary_rpy2[match_idx2][:, 0], ukf_rpy2[match_idx2][:, 0], gt_rpy2[match_idx2][:, 0]))
    y_or_array2 = np.vstack((eskf_rpy2[:, 1], complementary_rpy2[match_idx2][:, 1], ukf_rpy2[match_idx2][:, 1], gt_rpy2[match_idx2][:, 1]))
    z_or_array2 = np.vstack((eskf_rpy2[:, 2], complementary_rpy2[match_idx2][:, 2], ukf_rpy2[match_idx2][:, 2], gt_rpy2[match_idx2][:, 2]))


    ### Get the data we need -- Dataset 3
    ukf_data3, ukf_timestamp3, ukf_rpy3 = load_stuff.load_ukf(dataset3)
    gt_data3, gt_timestamp3, gt_position3, gt_velocity3, gt_rpy3 = load_stuff.load_gt(dataset3)
    eskf_data3, eskf_timestamp3, eskf_position3, eskf_velocity3, eskf_rpy3 = load_stuff.load_eskf(dataset3)
    complementary_data3, complementary_timestamp3, complementary_rpy3 = load_stuff.load_complementary(dataset3)

    # Match the timesteps of the ESKF with gt in order to make a perceptron of the positions
    match_idx3 = []
    for i in range(len(eskf_timestamp3)):
        match_idx3.append(np.argmin(np.abs(gt_timestamp3 - eskf_timestamp3[i])))

    # Make a numpy array of all of the filters x,y,z positions
    x_or_array3 = np.vstack((eskf_rpy3[:, 0], complementary_rpy3[match_idx3][:, 0], ukf_rpy3[match_idx3][:, 0], gt_rpy3[match_idx3][:, 0]))
    y_or_array3 = np.vstack((eskf_rpy3[:, 1], complementary_rpy3[match_idx3][:, 1], ukf_rpy3[match_idx3][:, 1], gt_rpy3[match_idx3][:, 1]))
    z_or_array3 = np.vstack((eskf_rpy3[:, 2], complementary_rpy3[match_idx3][:, 2], ukf_rpy3[match_idx3][:, 2], gt_rpy3[match_idx3][:, 2]))


    ### Stack all of the datasets together to train
    x_or_array = np.hstack((x_or_array1, x_or_array2, x_or_array3)).transpose()
    y_or_array = np.hstack((y_or_array1, y_or_array2, y_or_array3)).transpose()
    z_or_array = np.hstack((z_or_array1, z_or_array2, z_or_array3)).transpose()

    # Pass it into the perceptron!
    # pos_perceptron(x_pos_array, y_pos_array, z_pos_array)
    orientation_perceptron(x_or_array, y_or_array, z_or_array, dense) # UNCOMMENT IN ORDER TO TRAIN THE PERCEPTRON

if __name__ == "__main__":
    print("\nRun python main.py --help to see how to provide command line arguments\n\n")
    main()
    