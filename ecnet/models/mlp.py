#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/models/mlp.py
# v.3.2.3
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the "MultilayerPerceptron" (feed-forward neural network) class
#

from os import environ
from re import compile, IGNORECASE

from h5py import File
from numpy import array, string_, zeros

from tensorflow import config, Tensor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ecnet.utils.logging import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config.experimental_run_functions_eagerly(True)
H5_EXT = compile(r'.*\.h5', flags=IGNORECASE)


def check_h5(filename: str):
    ''' Ensures a given filename has an `.h5` extension

    Args:
        filename (str): filename to check
    '''

    if H5_EXT.match(filename) is None:
        raise ValueError(
            'Invalid filename/extension, must be `.h5`: {}'.format(
                filename
            )
        )


class MultilayerPerceptron(Model):

    def __init__(self, filename: str = 'model.h5'):
        ''' MultilayerPerceptron: Feed-forward neural network; variable number
        of layers, variable size/activations of layers; handles training,
        saving/loading of models

        Args:
            filename (str): filename/path for the model (default: `model.h5`)
        '''

        super(MultilayerPerceptron, self).__init__()
        check_h5(filename)
        self._filename = filename
        self._layers = []

    def add_layer(self, num_neurons: int, activation: str,
                  input_dim: int = None):
        ''' add_layer: adds a layer to the MLP; layers are added sequentially;
        first layer must have input dimensionality specified

        Args:
            num_neurons (int): number of neurons in the layer
            activation (str): activation function used by the layer; refer to
               https://www.tensorflow.org/api_docs/python/tf/keras/activations
               for usable activation functions
            input_dim (int): specify input dimensionality for the first layer;
               all other layers depend on previous layers' size (number of
               neurons), should be kept as `None` (default value)
        '''

        if len(self._layers) == 0:
            if input_dim is None:
                raise ValueError('First layer must have input_dim specified')

        self._layers.append(Dense(
            units=num_neurons,
            activation=activation,
            input_shape=(input_dim,)
        ))

    def call(self, x: Tensor) -> Tensor:
        ''' call: used by Model.fit (parent) to perform feed-forward operations

        Args:
            x (tf.Tensor): data fed into first layer

        Returns:
            tf.Tensor: data resulting from last layer
        '''

        for layer in self._layers:
            x = layer(x)
        return x

    def fit(self, l_x: array, l_y: array, v_x: array = None, v_y: array = None,
            epochs: int = 1500, lr: float = 0.001, beta_1: float = 0.9,
            beta_2: float = 0.999, epsilon: float = 0.0000001,
            decay: float = 0.0, v: int = 0, batch_size: int = 32,
            patience: int = 128) -> tuple:
        ''' fit: trains model using supplied data; may supply additional data
        to use as validation set (determines learning cutoff); hyperparameters
        for Adam optimization function, batch size, patience (if validating)
        may be specified

        Args:
            l_x (np.array): learning input data; each sub-iterable is a sample
            l_y (np.array): learning target data; each sub-iterable is a sample
            v_x (np.array): validation input data (`None` for no validation)
            v_y (np.array): validation target data (`None` for no validation)
            epochs (int): number of training iterations, max iterations if
                performing validation
            lr (float): learning rate of Adam optimization fn
            beta_1 (float): first moment estimate of Adam optimization fn
            beta_2 (float): second moment estimate of Adam optimization fn
            epsilon (float): number to prevent division by zero in Adam fn
            decay (float): decay of learning date in Adam optimization fn
            v (int): whether Model.fit (parent) is verbose (1 True, 0 False)
            batch_size (int): size of each training batch
            patience (int): maximum number of epochs to wait before better
                validation loss found; if not found, training terminates, best
                weights restored

        Returns:
            tuple: (list: learn losses, list: valid losses); each list equal
                length; each list element represents loss at corresponding
                epoch
        '''

        self.compile(optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,
                                    epsilon=epsilon,
                                    decay=decay),
                     loss=MeanSquaredError())

        if v_x is not None and v_y is not None:

            callback = EarlyStopping(monitor='val_loss', patience=patience,
                                     restore_best_weights=True)
            history = super().fit(l_x, l_y, batch_size=batch_size,
                                  epochs=epochs, verbose=v,
                                  callbacks=[callback],
                                  validation_data=(v_x, v_y))
            return (history.history['loss'], history.history['val_loss'])

        else:

            history = super().fit(l_x, l_y, batch_size=batch_size,
                                  epochs=epochs, verbose=v)
            return (history.history['loss'], [None for _ in range(epochs)])

    def use(self, x: array) -> array:
        ''' use: uses the model to predict values for supplied data

        Args:
            x (np.array): input data to predict for

        Returns:
            np.array: predicted values
        '''

        return self.predict(x)

    def save(self, filename: str = None):
        ''' save: saves the model weights, architecture to either the filename/
        path specified when object was created, or new, supplied filename/path

        Args:
            filename (str): new filepath if different than init filename/path
        '''

        if filename is None:
            filename = self._filename
        check_h5(filename)
        self.save_weights(filename, save_format='h5')
        input_size = self.layers[0].get_config()['batch_input_shape'][1]
        layer_sizes = [l.get_config()['units'] for l in self.layers]
        layer_activ = [l.get_config()['activation'] for l in self.layers]
        with File(filename, 'a') as hf:
            hf['mlp_input_size'] = input_size
            hf['mlp_layer_sizes'] = layer_sizes
            hf['mlp_layer_activ'] = string_(layer_activ)
        hf.close()
        logger.log('debug', 'Model saved to {}'.format(filename),
                   call_loc='MLP')

    def load(self, filename: str = None):
        ''' load: loads a saved model, restoring the architecture/weights;
        loads from filename/path specified during object initialization,
        unless new filename/path specified

        Args:
            filename (str): new filepath if different than init filename/path
        '''

        if filename is None:
            filename = self._filename
        with File(filename, 'r') as hf:
            input_size = hf.get('mlp_input_size').value
            layer_sizes = hf.get('mlp_layer_sizes').value
            layer_activ = hf.get('mlp_layer_activ').value
        hf.close()
        self.add_layer(layer_sizes[0], layer_activ[0].decode('ascii'),
                       input_size)
        for idx, layer in enumerate(layer_sizes[1:]):
            self.add_layer(layer, layer_activ[idx].decode('ascii'))
        self.build(input_shape=(None, input_size))
        self.load_weights(filename)
        logger.log('debug', 'Model loaded from {}'.format(filename),
                   call_loc='MLP')
