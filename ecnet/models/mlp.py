#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/models/mlp.py
# v.3.1.1
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the "MultilayerPerceptron" (feed-forward neural network) class
#

# Stdlib imports
from re import compile, IGNORECASE
from os import devnull, environ
import sys

# 3rd party imports
from tensorflow import get_default_graph, logging
from numpy import array
stderr = sys.stderr
sys.stderr = open(devnull, 'w')
from keras.backend import clear_session, reset_uids
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.metrics import mae
from keras.models import load_model, Sequential
from keras.optimizers import Adam
sys.stderr = stderr

# ECNet imports
from ecnet.utils.logging import logger

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity(logging.ERROR)

H5_EXT = compile(r'.*\.h5', flags=IGNORECASE)


class MultilayerPerceptron:

    def __init__(self, filename: str='model.h5'):
        '''MultilayerPerceptron object: fits neural network to supplied inputs
        and targets

        Args:
            filename (str): path to model save location (.h5 extension)
        '''

        if H5_EXT.match(filename) is None:
            raise ValueError(
                'Invalid filename/extension, must be `.h5`: {}'.format(
                    filename
                )
            )
        self._filename = filename
        clear_session()
        self._model = Sequential(name=filename.lower().replace('.h5', ''))

    def add_layer(self, num_neurons: int, activation: str,
                  input_dim: int=None):
        '''Adds a fully-connected layer to the model

        Args:
            num_neurons (int): number of neurons for the layer
            activation (str): activation function for the layer (see Keras
                activation function documentation)
            input_dim (int): if not None (input layer), specifies input
                dimensionality
        '''

        self._model.add(Dense(
            units=num_neurons,
            activation=activation,
            input_shape=(input_dim,)
        ))

    def fit(self, l_x: array, l_y: array, v_x: array=None, v_y: array=None,
            epochs: int=1500, lr: float=0.001, beta_1: float=0.9,
            beta_2: float=0.999, epsilon: float=0.0000001, decay: float=0.0,
            v: int=0):
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
        '''

        self._model.compile(
            loss=mean_squared_error,
            optimizer=Adam(
                lr=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                decay=decay
            ),
            metrics=[mae]
        )

        if v_x is not None and v_y is not None:
            valid_mae_lowest = self._model.evaluate(v_x, v_y, verbose=v)[1]
            steps = int(epochs / 250)
            for e in range(steps):
                h = self._model.fit(
                    l_x,
                    l_y,
                    validation_data=(v_x, v_y),
                    epochs=250,
                    verbose=v
                )
                valid_mae = h.history['val_mean_absolute_error'][-1]
                if valid_mae < valid_mae_lowest:
                    valid_mae_lowest = valid_mae
                elif valid_mae > (valid_mae_lowest + 0.05 * valid_mae_lowest):
                    logger.log('debug', 'Validation cutoff after {} epochs'
                               .format(e * 250), call_loc='MLP')
                    return

        else:
            self._model.fit(
                l_x,
                l_y,
                epochs=epochs,
                verbose=v
            )
        logger.log('debug', 'Training complete after {} epochs'.format(epochs),
                   call_loc='MLP')

    def use(self, x: array) -> array:
        '''Uses neural network to predict values for supplied data

        Args:
            x (numpy.array): input data to predict for

        Returns
            numpy.array: predictions
        '''

        with get_default_graph().as_default():
            return self._model.predict(x)

    def save(self, filename: str=None):
        '''Saves neural network to .h5 file

        filename (str): if None, uses MultilayerPerceptron._filename;
            otherwise, saves to this file
        '''

        if filename is None:
            filename = self._filename
        if H5_EXT.match(filename) is None:
            raise ValueError(
                'Invalid filename/extension, must be `.h5`: {}'.format(
                    filename
                )
            )
        self._model.save(filename)
        logger.log('debug', 'Model saved to {}'.format(filename),
                   call_loc='MLP')

    def load(self, filename: str=None):
        '''Loads neural network from .h5 file

        Args:
            filename (str): path to .h5 model file
        '''

        if filename is None:
            filename = self._filename
        self._model = load_model(filename)
        logger.log('debug', 'Model loaded from {}'.format(filename),
                   call_loc='MLP')
