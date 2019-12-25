#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/models/mlp.py
# v.3.2.3
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the "MultilayerPerceptron" (feed-forward neural network) class
#

# Stdlib imports
from re import compile, IGNORECASE

# 3rd party imports
from numpy import arange, array, asarray
from numpy.random import choice
from torch import load, relu, save, sigmoid, stack, tensor
from torch.nn import Linear, Module, MSELoss
from torch.optim import Adam

# ECNet imports
from ecnet.utils.logging import logger

ECNET_EXT = compile(r'.*\.ecnet', flags=IGNORECASE)


def check_ext(filename: str):
    '''Ensures the supplied filename has a `.ecnet` extension

    Args:
        filename (str): supplied filename
    '''

    if ECNET_EXT.match(filename) is None:
        raise ValueError('Invalid filename/extension, must be `.ecnet`: {}'
                         .format(filename))


class MultilayerPerceptron(Module):

    def __init__(self, filename: str='model.ecnet'):
        '''MultilayerPerceptron object: fits neural network to supplied inputs
        and targets

        Args:
            filename (str): path to model save location (.ecnet extension)
        '''

        super(MultilayerPerceptron, self).__init__()
        check_ext(filename)
        self._filename = filename
        self._layers = []

    def add_layer(self, num_neurons: int, activation: str,
                  input_dim: int=None):
        '''Adds a fully-connected layer to the model

        Args:
            num_neurons (int): number of neurons for the layer
            activation (str): activation function for the layer (`sigmoid`,
                `relu`, `linear`)
            input_dim (int): if not None (input layer), specifies input
                dimensionality
        '''

        if activation == 'sigmoid' or activation == sigmoid:
            activation = sigmoid
        elif activation == 'relu' or activation == relu:
            activation = relu
        elif activation == 'linear' or activation is None:
            activation = None
        else:
            raise ValueError('Unknown activation: {}'.format(activation))
        if input_dim is not None:
            self._layers.append((
                input_dim, num_neurons, activation
            ))
        else:
            if len(self._layers) == 0:
                raise AttributeError('Must include input dimensionality for '
                                     'first layer')
            self._layers.append((
                self._layers[-1][1], num_neurons, activation
            ))
        setattr(self, '_l{}'.format(len(self._layers) - 1), Linear(
            self._layers[-1][0], self._layers[-1][1]
        ))

    def fit(self, l_x: array, l_y: array, v_x: array=None, v_y: array=None,
            epochs: int=1500, lr: float=0.001, beta_1: float=0.9,
            beta_2: float=0.999, epsilon: float=0.0000001, decay: float=0.0,
            v: int=0, batch_size: int=32, patience: int=32) -> list:
        '''Fits neural network to supplied inputs and targets

        Args:
            l_x (numpy.array): learning input data
            l_y (numpy.array): learning target data
            v_x (numpy.array): if not None, periodic validation is performed w/
                these inputs
            v_y (numpy.array): if not None, periodic validation is performed w/
                these targets
            epochs (int): number of learning epochs if not validating, maximum
                number of learning epochs if performing periodic validation
            lr (float): learning rate for Adam optimizer
            beta_1 (float): beta_1 value for Adam optimizer
            beta_2 (float): beta_2 value for Adam optimizer
            epsilon (float): epsilon value for Adam optimizer
            decay (float): learning rate decay for Adam optimizer
            v (int): verbose training, `0` for no printing, `1` for printing
            batch_size (int): number of learning samples per batch
            patience (int): if performing periodic validation, will wait this
                many epochs for a better validation loss; will terminate
                training if better loss not found

        Returns:
            list: list of tuples, (train loss, valid loss) at each epoch; if
                not performing periodic validation, index 1 of tuple is None
        '''

        criterion = MSELoss()
        optimizer = Adam(
            self.parameters(),
            lr,
            (beta_1, beta_2),
            epsilon,
            decay
        )

        losses = []
        trained_epochs = epochs

        if v_x is not None and v_y is not None:
            best_model_params = self.state_dict()
            best_valid_loss = None
            valid_count = 0
            for e in range(epochs):
                train_x, train_y = self._batch(l_x, l_y, batch_size)
                valid_x, valid_y = self._batch(v_x, v_y, batch_size)
                optimizer.zero_grad()
                train_output = self(train_x)
                train_loss = criterion(train_output, train_y)
                train_loss.backward()
                optimizer.step()
                valid_output = self(valid_x)
                valid_loss = criterion(valid_output, valid_y)
                if v == 1:
                    logger.log('debug', 'Epoch {} | Training Loss {} | '
                               'Validation Loss {}'.format(e + 1, train_loss,
                               valid_loss), call_loc='MLP')
                losses.append((train_loss.item(), valid_loss.item()))
                if best_valid_loss is None or valid_loss < best_valid_loss:
                    best_model_params = self.state_dict()
                    best_valid_loss = valid_loss
                    valid_count = 0
                else:
                    valid_count += 1
                    if valid_count >= patience:
                        trained_epochs = e + 1
                        break
            self.load_state_dict(best_model_params)

        else:
            for e in range(epochs):
                train_x, train_y = self._batch(l_x, l_y, batch_size)
                optimizer.zero_grad()
                output = self(train_x)
                loss = criterion(output, train_y)
                if v == 1:
                    logger.log('debug', 'Epoch {} | Training Loss {}'
                               .format(e + 1, loss), call_loc='MLP')
                losses.append((loss.item(), None))
                loss.backward()
                optimizer.step()

        logger.log('debug', 'Training complete after {} epochs'
                   .format(trained_epochs), call_loc='MLP')
        return losses

    def use(self, x: array) -> array:
        '''Uses neural network to predict values for supplied data

        Args:
            x (numpy.array): input data to predict for

        Returns
            numpy.array: predictions
        '''

        return asarray(self(tensor(x).float()).tolist())

    def save(self, filename: str=None):
        '''Saves neural network to specified file

        filename (str): if None, uses MultilayerPerceptron._filename;
            otherwise, saves to this file
        '''

        if filename is None:
            filename = self._filename
        check_ext(filename)
        save({
            'layers': self._layers,
            'state_dict': self.state_dict()
        }, filename)
        logger.log('debug', 'Model saved to {}'.format(filename),
                   call_loc='MLP')

    def load(self, filename: str=None):
        '''Loads neural network from specified file

        Args:
            filename (str): if None, uses MultilayerPerceptron._filename;
                otherwise, loads from this file
        '''

        if filename is None:
            filename = self._filename
        check_ext(filename)
        file_contents = load(filename)
        layers = file_contents['layers']
        self._layers = []
        for idx, l in enumerate(layers):
            if idx == 0:
                self.add_layer(l[1], l[2], l[0])
            else:
                self.add_layer(l[1], l[2])
        self.load_state_dict(file_contents['state_dict'])
        logger.log('debug', 'Model loaded from {}'.format(filename),
                   call_loc='MLP')

    def forward(self, x: tensor) -> tensor:
        '''Used by torch.nn.Module to perform forward pass through neural
        network

        Args:
            x (torch.tensor): data to pass through neural network

        Returns:
            tensor: final layer's output
        '''

        for idx, layer in enumerate(self._layers):
            if layer[2] is None:
                x = getattr(self, '_l{}'.format(idx))(x)
            else:
                x = layer[2](getattr(self, '_l{}'.format(idx))(x))
        return x

    @staticmethod
    def _batch(x: array, y: array, batch_size: int) -> array:
        '''Returns a batch with specified size given input/target data

        Args:
            x (np.array): input data
            y (np.array): target data, same # samples as input data
            batch_size (int): size of the batch

        Returns:
            tuple: (np.array, np.array), each array of size batch_size
        '''

        if batch_size >= len(x):
            x_b = tensor(x).float()
            y_b = tensor(y).float()
        else:
            idxs = choice(arange(len(x)), batch_size, replace=False)
            x_b = tensor(x[idxs]).float()
            y_b = tensor(y[idxs]).float()
        return (x_b, y_b)
