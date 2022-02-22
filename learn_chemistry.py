from collections import OrderedDict
import sys
import os

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import torch
from torch import nn
from torch.functional import F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

class ArrheniusDataset(Dataset):
    def __init__(self):
        self._load_data('arrhenius.dataset')
        x = self.df[['reactant1', 'reactant2', 'product1', 'product2']].values
        y = self.df[['A', 'b', 'Ea']].values
        self._x = torch.tensor(x)
        self._y = torch.tensor(y)
        self.standardize()

    @property
    def num_inputs(self):
        return np.product(self.x.shape[1:])

    @property
    def num_outputs(self):
        return self.y.shape[-1]

    def standardize(self):
        """Standardize the inputs and outputs

        The inputs are categorial integers representing the appropriate specie
        participating in the reaction. These are converted to one hot encoded
        vectors.

        The outputs are floating point numbers that need to be rescaled to be
        able to make sense of them. The pre-exponential factor has a large
        variation in the order of magnitude so a logarithm is taken before
        normalizing.
        """
        # Do the one hot encoding
        self.x = F.one_hot(self._x)
        # Create a copy of the outputs
        self.y = self._y.detach().clone()
        # Normalize the outputs
        logA = np.log(self.y[:, 0])
        self.y[:, 0] = logA
        self._minimum = self.y.min(dim=0).values
        self._maximum = self.y.max(dim=0).values
        self.y = (self.y - self._minimum) / (self._maximum - self._minimum)

    def _unstandardize(self, y):
        """Unstandardize the outputs

        The outputs of the neural network are standardized. This provides a
        function to convert them back to their regular units.
        """
        # Undo the standardization
        y = y * (self._maximum - self._minimum) + self._minimum
        # Undo the logarithm of the pre-exponential factor
        y[:, 0] = np.exp(y[:, 0])
        #TODO: This doesn't work as expected in the first column
        return y

    def _load_data(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                self.df, self._species = pickle.load(f)
        else:
            # The pickle file doesn't exist - preprocess data and store it
            self._preprocess_data(filename, filename.replace('.dataset', '.csv'))

    def _preprocess_data(self, pickle_filename, csv_filename):
        self.df = pd.read_csv(csv_filename)
        df_species = self.df[['reactant1', 'reactant2', 'product1', 'product2']]
        self._species = pd.concat([df_species[col] for col in df_species.columns]).unique()
        f = lambda x: np.where(self._species == x)[0][0]
        df_species = df_species.applymap(f)
        self.df[df_species.columns] = df_species
        with open(pickle_filename, 'wb') as f:
            pickle.dump((self.df, self._species), f)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ArrheniusNet(nn.Module):
    """Create a network to predict Arrhenius rate parameters

    This network attempts to learn Arrhenius rate parameters from reactants and
    products. This network only works for reactions with two reactants and two
    products. The reactants and products are provided as one-hot encoded
    vectors.
    """
    def __init__(self, num_layers, num_inputs, nodes_per_layer, num_outputs):
        super().__init__()
        layers = OrderedDict()
        # Add the input layer
        layers['flatten'] = nn.Flatten()
        layers['input'] = nn.Linear(num_inputs, nodes_per_layer)
        layers['relu0'] = nn.ReLU()
        # Add the hidden layers
        for i in range(num_layers):
            layers[f'lin{i+1}'] = nn.Linear(nodes_per_layer, nodes_per_layer)
            layers[f'relu{i+1}'] = nn.ReLU()
        # Add the output layer
        layers['output'] = nn.Linear(nodes_per_layer, num_outputs)
        layers['reluout'] = nn.ReLU()
        # Create the network
        self.linear_relu_stack = nn.Sequential(layers)
        # Store the model parameters to help create a project name later
        self._num_layers = num_layers
        self._num_inputs = num_inputs
        self._nodes_per_layer = nodes_per_layer
        self._num_outputs = num_outputs

    def forward(self, x):
        return self.linear_relu_stack(x)

    @property
    def project_name(self):
        return (
                f'ArrheniusNet'
                + f'_nl{self._num_layers}'
                + f'_in{self._num_inputs}'
                + f'_npl{self._nodes_per_layer}'
                + f'_out{self._num_outputs}'
                )

class TrainingManager:

    def __init__(self, model=None, loss_fn=None, optimizer=None, training_dataloader=None, testing_dataloader=None, load_project=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_dataloader = training_dataloader
        self.testing_dataloader = testing_dataloader
        self.epoch = 0
        self.training_loss = []
        self.testing_loss = []
        self.project_name = model.project_name
        if load_project:
            self.load(model.project_name)

    def _train_one_epoch(self):
        size = len(self.training_dataloader.dataset)
        num_batches = len(self.training_dataloader)
        # Begin training
        running_loss = 0.
        self.model.train()
        for batch, (X, y) in enumerate(self.training_dataloader):
            X, y = X.to(device).float(), y.to(device).float()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        # Compute the loss per record to compare it with the testing loss
        self.training_loss.append(running_loss / size)
        self.epoch += 1

    def _test(self, print_comparison=False):
        size = len(self.testing_dataloader.dataset)
        num_batches = len(self.testing_dataloader)
        self.model.eval()
        running_loss = 0.
        with torch.no_grad():
            for X, y in self.testing_dataloader:
                X, y = X.to(device).float(), y.to(device).float()
                pred = self.model(X)
                if print_comparison:
                    print('Comparison (true vs estimated):')
                    for true, estimate in zip(y, pred):
                        print(f'  {true} - {estimate} (diff = {true - estimate})')
                running_loss += self.loss_fn(pred, y).item()
        # Compute the loss per record to compare it with the training loss
        self.testing_loss.append(running_loss / size)

    def train_loop(self, epochs):
        for i in range(epochs):
            self._train_one_epoch()
            self._test()
            print(f'Epoch {self.epoch:03d} - Training {self.training_loss[-1]:.3e} | Testing {self.testing_loss[-1]:.3e}')

    def save(self):
        torch.save({
            'epochs': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn': self.loss_fn,
            'training_loss': self.training_loss,
            'testing_loss': self.testing_loss,
            }, self.project_name + '.pth')

    def load(self, project_name):
        if os.path.isfile(self.project_name + '.pth'):
            checkpoint = torch.load(self.project_name + '.pth')
            print(f'Reading {self.project_name}.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('Previously trained model weights state_dict loaded.')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Previously trained optimizer state_dict loaded.')
            self.loss_fn = checkpoint['loss_fn']
            print('Trained model loss function loaded.')
            self.epoch = checkpoint['epochs']
            print(f'Model has been trained for {self.epoch} epochs')
            self.training_loss = checkpoint['training_loss']
            self.testing_loss = checkpoint['testing_loss']
            print(f'Last loss: training {self.training_loss[-1]:.3e} | Testing {self.testing_loss[-1]:.3e}')

#TODO: START HERE
# Create an instance of the TrainingManager
# Try to train, save, load, train
# Check how to initialize the learning objects
# Data objects can still be initialized outside

def initialize_learning_objects(ads, num_layers=5, nodes_per_layer=128):
    ann = ArrheniusNet(
            num_layers=num_layers,
            num_inputs=ads.num_inputs,
            nodes_per_layer=nodes_per_layer,
            num_outputs=ads.num_outputs,
            )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ann.parameters(), lr=1e-3)
    return ann, optimizer, loss_fn

def initialize_data_objects(training_percent=0.8, batch_size=64):
    ads = ArrheniusDataset()
    # Split into training and testing data
    train_size = int(0.8 * len(ads))
    test_size = len(ads) - train_size
    train_dataset, test_dataset = random_split(ads, [train_size, test_size])
    # Create the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return ads, train_dataloader, test_dataloader

def main(epochs=20, resume=False):
    ads, train_dataloader, test_dataloader = initialize_data_objects()
    ann, optimizer, loss_fn = initialize_learning_objects(ads)
    tm = TrainingManager(ann, loss_fn, optimizer, train_dataloader, test_dataloader, load_project=resume)
    tm.train_loop(epochs)
    tm.save()
    return tm
