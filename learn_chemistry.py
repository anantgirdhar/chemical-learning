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
    def __init__(self, num_layers, num_inputs, nodes_per_layer, num_outputs=3):
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

    def forward(self, x):
        return self.linear_relu_stack(x)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(f'dataloader {size = }')
    # We want to print the loss information about 10 times per training run
    # Compute how many batches to print the loss after
    print_every = int(num_batches / 10)
    # Begin training
    model.train()
    current = 0  # the current
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val, current = loss.item(), current + len(X)
        if batch % print_every == 0:
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    # print(f'Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    print(f'Test error: \n Avg loss per batch: {test_loss:>8f} \n')

def main(epochs=20):
    ads = ArrheniusDataset()
    # Split into training and testing data
    train_size = int(0.8 * len(ads))
    test_size = len(ads) - train_size
    train_dataset, test_dataset = random_split(ads, [train_size, test_size])
    # Create the DataLoaders
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # Create the Neural Network
    ann = ArrheniusNet(
            num_layers=2,
            num_inputs=ads.num_inputs,
            nodes_per_layer=64,
            num_outputs=ads.num_outputs,
            )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ann.parameters(), lr=1e-3)
    for _ in range(epochs):
        train(train_dataloader, ann, loss_fn, optimizer)
        test(test_dataloader, ann, loss_fn)
    return ann, optimizer, loss_fn, train_dataloader, test_dataloader
