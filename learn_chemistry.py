from collections import OrderedDict
import sys
import os

import matplotlib.pyplot as plt
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
    def __init__(self, include_atom_counts=False):
        self._load_data('arrhenius.dataset')
        features = ['reactant1', 'reactant2', 'product1', 'product2']
        if include_atom_counts:
            atom_count_headers_prefix = ['r1', 'r2', 'p1', 'p2']
            atom_count_headers_suffix = ['C', 'H', 'O', 'N', 'Ar', 'He', 'Cl', 'S', 'I', 'Si']
            for sp in atom_count_headers_prefix:
                features.extend([sp + atom for atom in atom_count_headers_suffix])
        x = self.df[features].values
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

        The first four inputs are categorial integers representing the
        appropriate specie participating in the reaction. These are converted
        to one hot encoded vectors. All following inputs are integers
        representing atom counts.

        The outputs are floating point numbers that need to be rescaled to be
        able to make sense of them. The pre-exponential factor has a large
        variation in the order of magnitude so a logarithm is taken before
        normalizing.
        """
        # Do the one hot encoding
        x1 = F.one_hot(self._x[:, :4], num_classes=len(self.species_map))
        x1 = torch.flatten(x1, start_dim=1)
        self.x = torch.concat([x1, self._x[:, 4:]], dim=1)
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
                self.df, self.species_map = pickle.load(f)
        else:
            # The pickle file doesn't exist - preprocess data and store it
            self._preprocess_data(filename, filename.replace('.dataset', '.csv'))

    def _preprocess_data(self, pickle_filename, csv_filename):
        # Load the data
        self.df = pd.read_csv(csv_filename)
        if not os.path.isfile('species_map.p'):
            raise ValueError('species_map.p not found')
        with open('species_map.p', 'rb') as f:
            self.species_map = pickle.load(f)
        f = lambda x: self.species_map.index(x)
        # Transform the data
        df_species = self.df[['reactant1', 'reactant2', 'product1', 'product2']]
        df_species = df_species.applymap(f)
        self.df[df_species.columns] = df_species
        with open(pickle_filename, 'wb') as f:
            pickle.dump((self.df, self.species_map), f)

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
    def __init__(self, num_layers, num_inputs, nodes_per_layer, num_outputs, dropout=False):
        super().__init__()
        layers = OrderedDict()
        # Add the input layer
        # layers['flatten'] = nn.Flatten()
        layers['input'] = nn.Linear(num_inputs, nodes_per_layer)
        if dropout: layers['drop0'] = nn.Dropout(0.25)
        layers['relu0'] = nn.ReLU()
        # Add the hidden layers
        for i in range(num_layers):
            layers[f'lin{i+1}'] = nn.Linear(nodes_per_layer, nodes_per_layer)
            if dropout: layers[f'drop{i+1}'] = nn.Dropout(p=0.25)
            layers[f'relu{i+1}'] = nn.ReLU()
        # Add the output layer
        layers['output'] = nn.Linear(nodes_per_layer, num_outputs)
        if dropout: layers[f'dropout'] = nn.Dropout(p=0.25)
        layers['reluout'] = nn.ReLU()
        # Create the network
        self.linear_relu_stack = nn.Sequential(layers)
        # Store the model parameters to help create a project name later
        self._num_layers = num_layers
        self._num_inputs = num_inputs
        self._nodes_per_layer = nodes_per_layer
        self._num_outputs = num_outputs
        self._dropout = dropout

    def forward(self, x):
        return self.linear_relu_stack(x)

    @property
    def project_name(self):
        return (
                f'ArrheniusNet'
                + f'_in{self._num_inputs}'
                + f'_out{self._num_outputs}'
                + f'_nl{self._num_layers}'
                + f'_npl{self._nodes_per_layer}'
                + ('_dropout' if self._dropout else '')
                )

class TrainingManager:

    def __init__(self, model=None, loss_fn=None, optimizer=None, training_dataset=None, testing_dataset=None, load_project=True, batch_size=64):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        #TODO: Shuffle the data
        self.training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
        self.testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size)
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

    def get_evals(self, data='test', show=True):
        self.model.eval()
        if data == 'test':
            X = self.testing_dataset.dataset.x.float()
            y = self.testing_dataset.dataset.y.float()
        elif data == 'train':
            X = self.training_dataset.dataset.x.float()
            y = self.training_dataset.dataset.y.float()
        else:
            raise ValueError(f'Invalid choice for dataset: {data}')
        pred = self.model(X)
        # Find the best fit lines
        line1 = np.poly1d(np.polyfit(
            y[:, 0].detach().numpy(),
            pred[:, 0].detach().numpy(),
            1))
        line2 = np.poly1d(np.polyfit(
            y[:, 1].detach().numpy(),
            pred[:, 1].detach().numpy(),
            1))
        line3 = np.poly1d(np.polyfit(
            y[:, 2].detach().numpy(),
            pred[:, 2].detach().numpy(),
            1))
        # Plot everything
        plt.subplots(1, 3, sharey=True)
        _range = np.linspace(0, 1, 51)
        plt.subplot(131)
        plt.plot(y[:, 0].detach().numpy(), pred[:, 0].detach().numpy(), 'bx')
        plt.plot(_range, line1(_range), 'r-')
        plt.plot(_range, _range, 'k--')
        plt.title(line1)
        plt.xlabel('True value')
        plt.ylabel('Prediction')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.subplot(132)
        plt.plot(y[:, 1].detach().numpy(), pred[:, 1].detach().numpy(), 'bx')
        plt.plot(_range, line2(_range), 'r-')
        plt.plot(_range, _range, 'k--')
        plt.title(line2)
        plt.xlabel('True value')
        plt.xlim([0, 1])
        plt.subplot(133)
        plt.plot(y[:, 2].detach().numpy(), pred[:, 2].detach().numpy(), 'bx')
        plt.plot(_range, line3(_range), 'r-',)
        plt.plot(_range, _range, 'k--')
        plt.title(line3)
        plt.xlabel('True value')
        plt.xlim([0, 1])
        plt.suptitle(self.project_name + f' [{data}]')
        plt.savefig(self.project_name + '_pred.png')
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.cla()
            plt.clf()
            plt.close('all')

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
        print(f'\nTraining model {self.project_name}')
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
        else:
            print(f'Tried to load but file {self.project_name}.pth not found.')

    def plot_loss(self, log_scale=True, show=True):
        plt.figure()
        if log_scale:
            plt.semilogy(range(1, self.epoch+1), self.training_loss,'bo-', label='train loss')
            plt.semilogy(range(1, self.epoch+1), self.testing_loss,'rx-', label='test loss')
        else:
            plt.plot(range(1, self.epoch+1), self.training_loss,'bo-', label='train loss')
            plt.plot(range(1, self.epoch+1), self.testing_loss,'rx-', label='test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.title(self.project_name)
        plt.savefig(self.project_name + '_loss.png')
        if show:
            plt.show()
        else:
            plt.cla()
            plt.clf()
            plt.close('all')

def initialize_learning_objects(ads, num_layers=5, nodes_per_layer=128, include_dropout=False):
    ann = ArrheniusNet(
            num_layers=num_layers,
            num_inputs=ads.num_inputs,
            nodes_per_layer=nodes_per_layer,
            num_outputs=ads.num_outputs,
            dropout=include_dropout,
            )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ann.parameters(), lr=1e-3)
    return ann, optimizer, loss_fn

def initialize_data_objects(training_percent=0.8, batch_size=64, include_atom_counts=False):
    ads = ArrheniusDataset(include_atom_counts)
    # Split into training and testing data
    torch.manual_seed(0)
    train_size = int(0.8 * len(ads))
    test_size = len(ads) - train_size
    train_dataset, test_dataset = random_split(ads, [train_size, test_size])
    return ads, train_dataset, test_dataset

def main(num_layers=5, nodes_per_layer=128, epochs=40, include_dropout=False, include_atom_counts=False, resume=True):
    ads, train_dataset, test_dataset = initialize_data_objects(include_atom_counts=include_atom_counts)
    ann, optimizer, loss_fn = initialize_learning_objects(ads, num_layers, nodes_per_layer, include_dropout=include_dropout)
    tm = TrainingManager(ann, loss_fn, optimizer, train_dataset, test_dataset, load_project=resume)
    if not resume:
        tm.train_loop(epochs)
        tm.save()
    return tm

def run_multiple_models():
    for num_layers in [4, 5, 6]:
        for nodes_per_layer in [64, 128, 256]:
            for include_atom_counts in [True, False]:
                for include_dropout in [True, False]:
                    last_train_loss = 1e6
                    last_test_loss = 1e6
                    tm = main(num_layers=num_layers, nodes_per_layer=nodes_per_layer, epochs=20, include_dropout=include_dropout, include_atom_counts=include_atom_counts, resume=False)
                    tm.plot_loss(show=False)
                    tm.get_evals(show=False)

if __name__ == "__main__":
    run_multiple_models()
