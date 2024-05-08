import torch
import torchvision as thv
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
import pdb

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encode_linear1 = nn.Linear(196, 128)
        self.encode_linear2_1 = nn.Linear(128, 8)
        self.encode_linear2_2 = nn.Linear(128, 8)
        self.decode_linear1 = nn.Linear(8, 128)
        self.decode_linear2 = nn.Linear(128, 196)

    def encode(self, x):
        h1 = torch.tanh(self.encode_linear1(x))
        return self.encode_linear2_1(h1), self.encode_linear2_2(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.tanh(self.decode_linear1(z))
        return torch.sigmoid(self.decode_linear2(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 196))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def import_dataset(num_train_samples, num_val_samples):
    '''
    Function to import MNIST dataset
    Input:
        num_train_samples (int): number of samples per class for training data
        num_val_samples (int): number of samples per class for validation data
    Output:
        train_data (torch.Tensor): training data shape (num_classes*num_train_samples, 14*14)
        val_data (torch.Tensor): validation data shape (num_classes*num_val_samples, 14*14)
    '''
    ## downloading training and validation data
    train = thv.datasets.MNIST('./', download=True, train=True)
    val = thv.datasets.MNIST('./', download=True, train=False)

    x_train = train.data
    x_val = val.data
    y_train = train.targets
    y_val = val.targets

    num_images_train = num_train_samples
    num_images_val = num_val_samples
    train_data = torch.zeros((10*num_images_train, 14*14))
    val_data = torch.zeros((10*num_images_val, 14*14))

    for i in range(10):
        temp_idx = torch.where(y_train == i)[0]
        idx = np.random.choice(temp_idx, num_images_train, replace=False)
        for j in range(num_images_train):
            train_data[i*num_images_train+j] = torch.Tensor(cv2.resize(x_train[idx[j]].numpy(), (14,14)).flatten())
        
        temp_idx = torch.where(y_val == i)[0]
        idx = np.random.choice(temp_idx, num_images_val, replace=False)
        for j in range(num_images_val):
            val_data[i*num_images_val+j] = torch.Tensor(cv2.resize(x_val[idx[j]].numpy(), (14,14)).flatten())


    train_data = torch.where(train_data >= 128, 1., 0.)
    val_data = torch.where(val_data >= 128, 1., 0.)

    return train_data, val_data

def loss_function(pred1, pred2, x, mu, logvar):
    BCE1 = F.binary_cross_entropy(pred1, x.view(-1, 196), reduction='sum')
    BCE2 = F.binary_cross_entropy(pred2, x.view(-1, 196), reduction='sum')
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = (BCE1+BCE2)/2
    return BCE, KLD

def results_images_plot(model, val_data):
    '''
    Function to plot true images with model output images
    '''
    for i in range(8):
        fig, ax = plt.subplots(1,2)
        fig.tight_layout(w_pad=-0.5)
        idx = np.random.randint(0, val_data.size(0))
        pred, _, _ = model(val_data[idx].to(device))

        ax[0].imshow(val_data[idx].numpy().reshape((14,14)), cmap='gray')
        ax[1].imshow(pred.cpu().detach().numpy().reshape((14,14)), cmap='gray')

        plt.title('Original v/s Model generate Images')
        filename = "og_v_model_" + str(i+1) + ".png"
        plt.savefig(filename)
    # plt.show()

def generate_random_samples(model, mean, std, num_samples):
    '''
    Function to generate random samples from decoder of model using a gaussian distribution of latent space
    '''
    z = torch.normal(mean, std, size=(num_samples,8))
    pred = model.decode(z.to(device))
    for i in range(num_samples):
        plt.figure()
        plt.imshow(pred[i].cpu().detach().numpy().reshape((14,14)), cmap='gray')
        plt.title("Random Sample")
        filename = "random_sample_" + str(i+1) + ".png"
        plt.savefig(filename)
        # pdb.set_trace()

def metrics_plot(train_wt_updates, val_wt_updates, train_bce, train_kld, val_bce):
    '''
    Function to plot metrics
    Input:
        train_wt_updates (list)
        val_wt_updates (list)
        train_bce (list)
        train_kld (list)
        val_bce (list)
    '''
    plt.figure()
    plt.plot(train_bce, label="BCE {log(p_v)}")
    plt.title("BCE {log(p_v)}")
    plt.xlabel("Number of weight updates")
    plt.ylabel("BCE")
    plt.savefig("train_bce.png")

    plt.figure()
    plt.plot(train_kld, label="KLD")
    plt.title("KL Divergence")
    plt.xlabel("Number of weight updates")
    plt.ylabel("KLD")
    plt.savefig("train_kld.png")

    plt.figure()
    plt.plot(train_wt_updates, train_bce, "g", label="train BCE")
    plt.plot(val_wt_updates, val_bce, "r", label="val BCE")
    plt.title("BCE Train v/s Validation")
    plt.xlabel("Number of weight updates")
    plt.ylabel("BCE")
    plt.legend()
    plt.savefig("train_v_val_bce.png")



    
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data = import_dataset(1000, 10)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 100
    num_epochs = 100

    bce_list = []
    kld_list = []
    loss_list = []
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    verbose = False
    wt_updates = 0
    train_wt_update_ls = []
    val_wt_update_ls = []
    val_bce_list = []
    # pdb.set_trace()

    #training loop
    for epoch in tqdm(range(1, num_epochs+1)):

        model.train()
        train_loss = 0
        for _, data in enumerate(train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            pred1, mu, logvar = model(data)
            pred2, mu, logvar = model(data)
            bce, kld = loss_function(pred1, pred2, data, mu, logvar)
            loss = bce - kld
            loss.backward()
            train_loss += loss.item()
            bce_list.append(bce.item()/data.size(0))
            kld_list.append(kld.item()/data.size(0))
            optimizer.step()
            wt_updates += 1
            train_wt_update_ls.append(wt_updates)

        model.eval()
        for _, data in enumerate(val_dataloader):
            data = data.to(device)
            pred1, mu, logvar = model(data)
            pred2, mu, logvar = model(data)
            bce, kld = loss_function(pred1, pred2, data, mu, logvar)
            val_bce_list.append(bce.item()/data.size(0))
            val_wt_update_ls.append(wt_updates)

        if verbose: print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / train_data.size(0)))

    model.eval()

    ##all the plotting functions called
    
    results_images_plot(model, val_data)

    generate_random_samples(model, 0.0, 1.0, 5)
                
    metrics_plot(train_wt_update_ls, val_wt_update_ls, bce_list, kld_list, val_bce_list)