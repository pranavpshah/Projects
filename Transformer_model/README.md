# Attention-based Networks for Human Trajectory Prediction
This code base is for the final project of ESE 546. In this project, we implement the classical attention-based Transformer Network (TF) to predict the future trajectories of individuals in a scene. The project is inspired by the paper "Transformer Networks for Trajectory Forecasting". We use the TrajNet dataset which is a superset of diverse datasets. Training on this dataset ensures that our model generalizes and performs well in unseen situations. We build our model using different optimizer and scheduler techniques and analyze the one that gives us the best performance. We then perform extensive testing using the best model and present some quantitative and qualitative results. The results show that our best TF model is able to predict future pedestrian trajectories with an average error of ~45 cm.

## Dataset
The dataset is taken from TrajNet bank of datasets. The folder `dataset` contains all the data from TrajNet. Place the dataset folder in the working directory.

## Running the Training and Evaluation Loop
Execute the `train.py` script. This script will train the transformer model while performing validation after every epoch. This will also save all the plots from the training and validation in the current working directory. 

This script also performs trajectory predictions after the model had been trained and saves the prediction plots.
