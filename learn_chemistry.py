from collections import OrderedDict
import sys
import os

import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
from torch import nn
from torch.functional import F
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

SUMMARY_FILE='model_summary.csv'

class ArrheniusDataset(Dataset):
    def __init__(self, include_atom_counts=False, order_species=True):
        self._load_data('arrhenius.dataset')
        # If order species is False, then the order of the species does not
        # matter - duplicate the reactions with the species in various orders
        # so that the network can learn that
        if not order_species:
            self._add_swapped_versions()
        # extract the required variables
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
        self._ordered_species = order_species
        self._include_atom_counts = include_atom_counts

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

    def _add_swapped_versions(self):
        # Swap the reactants
        df_swapped_reactants = self.df.copy()
        df_swapped_reactants[['reactant1', 'reactant2']] = self.df[['reactant2', 'reactant1']]
        # Swap the products
        df_swapped_products = self.df.copy()
        df_swapped_products[['product1', 'product2']] = self.df[['product2', 'product1']]
        # Swap both reactants and products
        df_swapped_both = self.df.copy()
        df_swapped_both[['reactant1', 'reactant2', 'product1', 'product2']] = \
                self.df[['reactant2', 'reactant1', 'product2', 'product1']]
        # Append both
        self.df = pd.concat([self.df, df_swapped_reactants, df_swapped_products, df_swapped_both])

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
    def __init__(self, num_inputs, num_outputs, num_layers=None, nodes_per_layer=None, dropout=False, layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            # Create a list of layer sizes from the other inputs
            layer_sizes = [nodes_per_layer, ] * num_layers
            self._uniform_layers = True
        else:
            # Make sure that the given layer sizes are not all the same (in
            # which case, just revert to the uniform route)
            if len(set(layer_sizes)) == 1:
                # It's just uniform
                self._uniform_layers = True
                num_layers = len(layer_sizes)
                nodes_per_layer = layer_sizes[0]
            else:
                self._uniform_layers = False
        # Create the network now
        self._init_network(num_inputs, num_outputs, layer_sizes, dropout)
        # Store the model parameters to help create a project name later
        self._num_layers = num_layers
        self._num_inputs = num_inputs
        self._nodes_per_layer = nodes_per_layer
        self._num_outputs = num_outputs
        self._dropout = dropout
        self._layer_sizes = layer_sizes

    def _init_network(self, num_inputs, num_outputs, layer_sizes, dropout):
        layers = OrderedDict()
        # Add the input layer
        # layers['flatten'] = nn.Flatten()
        layers['input'] = nn.Linear(num_inputs, layer_sizes[0])
        if dropout: layers['drop0'] = nn.Dropout(0.25)
        layers['relu0'] = nn.ReLU()
        # Add the hidden layers
        for i in range(len(layer_sizes)-1):
            layers[f'lin{i+1}'] = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            if dropout: layers[f'drop{i+1}'] = nn.Dropout(p=0.25)
            layers[f'relu{i+1}'] = nn.ReLU()
        # Add the output layer
        layers['output'] = nn.Linear(layer_sizes[-1], num_outputs)
        if dropout: layers[f'dropout'] = nn.Dropout(p=0.25)
        layers['reluout'] = nn.ReLU()
        # Create the network
        self.linear_relu_stack = nn.Sequential(layers)

    def forward(self, x):
        return self.linear_relu_stack(x)

    @property
    def project_name(self):
        if self._uniform_layers:
            return (
                    f'ArrheniusNet'
                    + f'_in{self._num_inputs}'
                    + f'_out{self._num_outputs}'
                    + f'_nl{self._num_layers}'
                    + f'_npl{self._nodes_per_layer}'
                    + ('_dropout' if self._dropout else '')
                    )
        else:
            return (
                    f'ArrheniusNet'
                    + f'_in{self._num_inputs}'
                    + f'_out{self._num_outputs}'
                    + '_layers'
                    + '-'.join([str(s) for s in self._layer_sizes])
                    + ('_dropout' if self._dropout else '')
                    )

    def __repr__(self):
        if self._uniform_layers:
            return (
                    f'ArrheniusNet('
                    + f'num_inputs={self._num_inputs}'
                    + f', num_outputs={self._num_outputs}'
                    + f', num_layers={self._num_layers}'
                    + f', nodes_per_layer={self._nodes_per_layer}'
                    + (', dropout=True' if self._dropout else '')
                    + ')'
                    )
        else:
            return (
                    f'ArrheniusNet('
                    + f'num_inputs={self._num_inputs}'
                    + f', num_outputs={self._num_outputs}'
                    + (', dropout=True, ' if self._dropout else '')
                    + f', layer_sizes={self._layer_sizes}'
                    + ')'
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
        self.metrics = {
                'train_loss': [],
                'test_loss': [],
                'train_PC': [],
                'test_PC': [],
                'train_BFL': [],
                'test_BFL': [],
                }
        self.project_name = model.project_name
        if not self.training_dataset.dataset._ordered_species:
            self.project_name += '_unorderedsp'
        if self.training_dataset.dataset._include_atom_counts:
            self.project_name += '_withcounts'
        self._predictions_test = None
        self._predictions_train = None
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
        self.metrics['train_loss'].append(running_loss / size)
        self.epoch += 1
        # Reset the predictions
        self._predictions_test = None
        self._predictions_train = None

    @torch.no_grad()
    def _test(self):
        size = len(self.testing_dataloader.dataset)
        num_batches = len(self.testing_dataloader)
        running_loss = 0.
        self.model.eval()
        for X, y in self.testing_dataloader:
            X, y = X.to(device).float(), y.to(device).float()
            pred = self.model(X)
            running_loss += self.loss_fn(pred, y).item()
        # Compute the loss per record to compare it with the training loss
        self.metrics['test_loss'].append(running_loss / size)

    @torch.no_grad()
    def _update_test_train_predictions(self):
        self.model.eval()
        # Update the test predictions
        X = self.testing_dataset.dataset.x.float()
        self._predictions_test = self.model(X)
        # Update the training predictions
        X = self.training_dataset.dataset.x.float()
        self._predictions_train = self.model(X)

    def _compute_PC(self):
        if self._predictions_test is None or self._predictions_train is None:
            self._update_test_train_predictions()
        # Compute Pearson Coefficients for the testing data
        y_test = self.testing_dataset.dataset.y.float()
        pearson_coefficients = []
        for var in range(y_test.shape[1]):
            pearson_coefficients.append(np.corrcoef([
                y_test[:, var].detach().numpy(),
                self._predictions_test[:, var].detach().numpy()])[0][1])
        self.metrics['test_PC'].append(tuple(pearson_coefficients))
        # Compute Pearson Coefficients for the training data
        y_train = self.training_dataset.dataset.y.float()
        pearson_coefficients = []
        for var in range(y_train.shape[1]):
            pearson_coefficients.append(np.corrcoef([
                y_train[:, var].detach().numpy(),
                self._predictions_train[:, var].detach().numpy()])[0][1])
        self.metrics['train_PC'].append(tuple(pearson_coefficients))

    def _compute_best_fit_lines(self):
        if self._predictions_test is None or self._predictions_train is None:
            self._update_test_train_predictions()
        # Compute the best fit lines for the testing data
        y_test = self.testing_dataset.dataset.y.float().detach().numpy()
        self.metrics['test_BFL'].append([])
        for var in range(y_test.shape[1]):
            self.metrics['test_BFL'][-1].append(
                    np.poly1d(np.polyfit(
                        y_test[:, var],
                        self._predictions_test[:, var].detach().numpy(),
                        1
                        )))
        # Compute the best fit lines for the training data
        y_train = self.training_dataset.dataset.y.float().detach().numpy()
        self.metrics['train_BFL'].append([])
        for var in range(y_train.shape[1]):
            self.metrics['train_BFL'][-1].append(
                    np.poly1d(np.polyfit(
                        y_train[:, var],
                        self._predictions_train[:, var].detach().numpy(),
                        1
                        )))

    def _compute_metrics(self):
        self._compute_PC()
        self._compute_best_fit_lines()

    def train_loop(self, epochs):
        print(f'\nTraining model {self.project_name}')
        for i in range(epochs):
            self._train_one_epoch()
            self._test()
            self._compute_metrics()
            print(f'Epoch {self.epoch:03d} '
                  f'- Training {self.metrics["train_loss"][-1]:.3e} '
                  f'| Testing {self.metrics["test_loss"][-1]:.3e}'
                  )

    def save(self):
        torch.save({
            'epochs': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn': self.loss_fn,
            'metrics': self.metrics,
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
            self.metrics = checkpoint['metrics']
            print(f'Last loss: '
                  f'training {self.metrics["train_loss"][-1]:.3e} | '
                  f'Testing {self.metrics["test_loss"][-1]:.3e}'
                  )
        else:
            print(f'Tried to load but file {self.project_name}.pth not found.')

    def plot_scatters(self, show=True):
        if self._predictions_test is None or self._predictions_train is None:
            self._update_test_train_predictions()
        # Create aliases to the data for readability
        y_test = self.testing_dataset.dataset.y.float().detach().numpy()
        pred_test = self._predictions_test.detach().numpy()
        y_train = self.training_dataset.dataset.y.float().detach().numpy()
        pred_train = self._predictions_train
        num_vars = y_test.shape[1]
        # Create aliases to the metrics
        PC_test = self.metrics['test_PC'][-1]
        PC_train = self.metrics['train_PC'][-1]
        BFL_test = self.metrics['test_BFL'][-1]
        BFL_train = self.metrics['train_BFL'][-1]
        # Create two rows of plots
        # The top row will have the test scatter data
        # The bottom row will have the training scatter data
        plt.subplots(2, num_vars, sharey=True, sharex=True)
        _range = np.linspace(0, 1, 51)
        for var in range(num_vars):
            plt.subplot(231 + var)
            plt.plot(y_test[:, var], pred_test[:, var], 'bx')
            line = np.poly1d(BFL_test[var])
            plt.plot(_range, line(_range), 'r-')
            plt.plot(_range, _range, 'k--')
            plt.title(f'Test [{var=}]: {line}')
            plt.text(0.6, 0.05, f'PC = {PC_test[var]:.2f}', backgroundcolor='white', alpha=0.7)
            plt.subplot(231 + num_vars + var)
            plt.plot(y_train[:, var], pred_train[:, var], 'bx')
            line = np.poly1d(BFL_train[var])
            plt.plot(_range, line(_range), 'r-')
            plt.plot(_range, _range, 'k--')
            plt.title(f'Train [{var=}]: {line}')
            plt.text(0.6, 0.05, f'PC = {PC_train[var]:.2f}', backgroundcolor='white', alpha=0.7)
        plt.suptitle(self.project_name)
        plt.tight_layout()
        if show:
            plt.show()

    def plot_temporal_metrics(self, log_scale=True, show=True):
        plt.subplots(2, 1, sharex=True, figsize=(14, 10))
        _range = range(1, self.epoch+1)
        train_markers = ['bo', 'cs', 'kd']
        test_markers = ['rx', 'mv', 'y^']
        # Plot the loss
        plt.subplot(211)
        if log_scale:
            plt.semilogy(_range, self.metrics['train_loss'],'bo-', label='train loss')
            plt.semilogy(_range, self.metrics['test_loss'],'rx--', label='test loss')
        else:
            plt.plot(_range, self.metrics['train_loss'],'bo-', label='train loss')
            plt.plot(_range, self.metrics['test_loss'],'rx--', label='test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        # Plot the pearson correlation coefficients
        plt.subplot(223)
        PC_train = list(zip(*self.metrics['train_PC']))
        PC_test = list(zip(*self.metrics['test_PC']))
        for i in range(len(PC_train)):
            plt.plot(_range, PC_train[i], train_markers[i]+'-', label=f'PC {i} train')
            plt.plot(_range, PC_test[i], test_markers[i]+'--', label=f'PC {i} test')
        plt.xlabel('Epoch')
        plt.ylabel('PC')
        plt.legend(loc='best')
        # Plot the best fit line slopes
        plt.subplot(224)
        BFL_train = list(zip(*self.metrics['train_BFL']))
        BFL_test = list(zip(*self.metrics['test_BFL']))
        for i in range(len(BFL_train)):
            plt.plot(_range, [x[1] for x in BFL_train[i]], train_markers[i]+'-', label=f'BFL {i} train')
            plt.plot(_range, [x[1] for x in BFL_test[i]], test_markers[i]+'--', label=f'BFL {i} test')
        plt.xlabel('Epoch')
        plt.ylabel('BFL slopes')
        plt.legend(loc='best')
        plt.suptitle(self.project_name)
        plt.savefig(self.project_name + '_metrics.png')
        if show:
            plt.show()
        else:
            plt.cla()
            plt.clf()
            plt.close('all')

class CrossValidation:

    def __init__(self, ann_objects_fn, dataset, nfolds=10, batch_size=64, epochs=20):
        self.ann_objects_fn = ann_objects_fn
        self.dataset = dataset
        self.nfolds = nfolds
        self.batch_size = batch_size
        self.epochs = epochs
        self.folds = KFold(n_splits=nfolds, shuffle=True, random_state=42)
        self.folds = self.folds.split(range(len(dataset)))
        self.metrics = {f'fold{foldnum+1:02d}': None for foldnum in range(nfolds)}
        # To get the project name, create a throwaway model
        model, _, _ = self.ann_objects_fn()
        self.project_name = model.project_name
        if not dataset._ordered_species:
            self.project_name += '_unorderedsp'
        if dataset._include_atom_counts:
            self.project_name += '_withcounts'
        self.project_name += '_folds'

    def run(self):
        for fold, (trainidx, testidx) in enumerate(self.folds):
            model, optimizer, loss_fn = self.ann_objects_fn()
            tm = TrainingManager(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    training_dataset=Subset(self.dataset, trainidx),
                    testing_dataset=Subset(self.dataset, testidx),
                    load_project=False,
                    batch_size=self.batch_size)
            tm.train_loop(self.epochs)
            self.metrics[f'fold{fold+1:02d}'] = tm.metrics

    def save_metrics(self):
        with open(self.project_name + '.p', 'wb') as f:
            pickle.dump({
                'epochs': self.epochs,
                'metrics': self.metrics,
                }, f)

def initialize_learning_objects(ads, num_layers=5, nodes_per_layer=128, include_dropout=False, layer_sizes=None):
    ann = ArrheniusNet(
            num_inputs=ads.num_inputs,
            num_outputs=ads.num_outputs,
            num_layers=num_layers,
            nodes_per_layer=nodes_per_layer,
            dropout=include_dropout,
            layer_sizes=layer_sizes,
            )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ann.parameters(), lr=1e-3)
    return ann, optimizer, loss_fn

def initialize_data_objects(training_percent=0.8, batch_size=64, include_atom_counts=False, order_species=True):
    ads = ArrheniusDataset(include_atom_counts, order_species=order_species)
    # Split into training and testing data
    torch.manual_seed(0)
    train_size = int(0.8 * len(ads))
    test_size = len(ads) - train_size
    train_dataset, test_dataset = random_split(ads, [train_size, test_size])
    return ads, train_dataset, test_dataset

def main_train(num_layers=None, nodes_per_layer=None, epochs=40, include_dropout=False, layer_sizes=None, include_atom_counts=False, order_species=True, resume=True):
    ads, train_dataset, test_dataset = initialize_data_objects(include_atom_counts=include_atom_counts, order_species=order_species)
    ann, optimizer, loss_fn = initialize_learning_objects(ads, num_layers, nodes_per_layer, include_dropout=include_dropout, layer_sizes=layer_sizes)
    tm = TrainingManager(ann, loss_fn, optimizer, train_dataset, test_dataset, load_project=resume)
    if not resume:
        tm.train_loop(epochs)
        tm.save()
    return tm

def main_crossvalidate(num_layers=None, nodes_per_layer=None, epochs=40, include_dropout=False, layer_sizes=None, include_atom_counts=False, order_species=True, nfolds=10, batch_size=64):
    ads, _, _ = initialize_data_objects(include_atom_counts=include_atom_counts, order_species=order_species)
    ann_objects_fn = lambda: initialize_learning_objects(ads, num_layers, nodes_per_layer, include_dropout=include_dropout, layer_sizes=layer_sizes)
    cv = CrossValidation(
            ann_objects_fn,
            dataset=ads,
            nfolds=nfolds,
            batch_size=batch_size,
            epochs=epochs)
    cv.run()
    cv.save_metrics()
    return cv

def run_uniform_models(crossvalidation=False):
    num_layers = [2, 3, 4, 5, 6]
    nodes_per_layer = [16, 32, 64, 128, 256, 512]
    include_atom_counts = [True, False]
    include_dropout = [True, False]
    order_species = [True, False]
    for nl, npl, ic, dropout, order in it.product(
            num_layers, nodes_per_layer, include_atom_counts, include_dropout, order_species,
            ):
        last_train_loss = 1e6
        last_test_loss = 1e6
        if crossvalidation:
            cv = main_crossvalidate(
                    num_layers=nl, nodes_per_layer=npl,
                    epochs=20,
                    include_dropout=dropout,
                    include_atom_counts=ic,
                    order_species=order,
                    )
        else:
            tm = main_train(
                    num_layers=nl, nodes_per_layer=npl,
                    epochs=20,
                    include_dropout=dropout,
                    include_atom_counts=ic,
                    order_species=order,
                    resume=False,
                    )
            tm.plot_scatters(show=False)
            tm.plot_temporal_metrics(show=False)

def run_nonuniform_models(crossvalidation=False):
    architectures = [
            (16, 4, 8),
            (32, 8, 16),
            (64, 16, 32),
            (64, 32, 64),
            (128, 32, 64),
            (128, 64, 64),
            (256, 64, 128),
            (512, 128, 256),
            (32, 8, 16, 4),
            (64, 16, 32, 8),
            (64, 32, 64, 32),
            (128, 32, 64, 16),
            (128, 64, 64, 32),
            (256, 64, 128, 32),
            (512, 128, 256, 64),
            # (1000, 125, 500),
            # (1000, 125, 500, 50),
            # (2000, 500, 1000),
            (2000, 500, 1000, 100),
            ]
    # Sort the architectures in order of number of parameters
    def _arch_size(a):
        a = np.array(a)
        return sum(a[1:] * a[:-1]) + a[0] * 2000 + a[-1] * 3
    architectures = sorted(architectures, key=lambda a: _arch_size(a))
    include_atom_counts = [True, False]
    include_dropout = [True, False]
    order_species = [True, False]
    for arch, ic, dropout, order in it.product(
            architectures, include_atom_counts, include_dropout, order_species,
            ):
        last_train_loss = 1e6
        last_test_loss = 1e6
        if crossvalidation:
            cv = main_crossvalidate(
                    epochs=20,
                    include_dropout=dropout,
                    layer_sizes=arch,
                    include_atom_counts=ic,
                    order_species=order,
                    )
        else:
            tm = main_train(
                    epochs=20,
                    include_dropout=dropout,
                    layer_sizes=arch,
                    include_atom_counts=ic,
                    order_species=order,
                    )
            tm.plot_scatters(show=False)
            tm.plot_temporal_metrics(show=False)

if __name__ == "__main__":
    crossvalidation=False
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cv':
            crossvalidation = True
    run_uniform_models(crossvalidation=crossvalidation)
    run_nonuniform_models(crossvalidation=crossvalidation)
