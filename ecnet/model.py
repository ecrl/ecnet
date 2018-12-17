#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/error_utils.py
# v.2.0.0
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions necessary creating, training, saving, and reusing neural
# network models
#

# Stdlib imports
import pickle
from pickle import dump, load
from functools import reduce
from os import environ, mkdir, path
from random import uniform
from multiprocessing import current_process
from copy import deepcopy

# 3rd party imports
import tensorflow as tf
from tensorflow import add, global_variables_initializer, matmul, nn
from tensorflow import placeholder, random_normal, reset_default_graph
from tensorflow import Session, square, train, Variable
from numpy import asarray, sqrt as nsqrt

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def __linear_fn(n):
    '''Linear definition for activation function dictionary
    '''
    return n


ACTIVATION_FUNCTONS = {
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'softmax': nn.softmax,
    'linear': __linear_fn
}


class Layer:

    def __init__(self, size, act_fn):
        '''Layer object: contains layer size (number of neurons), activation
        function

        Args:
            size (int): number of neurons in the layer
            act_fn (str): 'relu', 'sigmoid', 'softmax', 'linear'
        '''

        assert type(size) is int, \
            'Invalid size type: {}'.format(type(size))
        assert act_fn in list(ACTIVATION_FUNCTONS.keys()), \
            'Invalid activation function: {}'.format(act_fn)

        self.size = size
        if act_fn not in ACTIVATION_FUNCTONS.keys():
            raise ValueError(
                'Unsupported activation function {}'.format(act_fn)
            )
        self.act_fn = ACTIVATION_FUNCTONS[act_fn]

    def __len__(self):
        '''
        Layer length = number of neurons = layer.size
        '''

        return self.size


class MultilayerPerceptron:

    def __init__(self, save_path=None, id=0):
        '''Neural network (multilayer perceptron) - contains methods for training,
        using, saving and opening neural network models

        Args:
            save_path (str): if not None, saves model to this path
        '''

        assert type(id) is int, 'Invalid id type: {}'.format(id)

        self.__weights = []
        self.__biases = []
        self._layers = []
        self._id = id
        if save_path is None:
            save_path = './tmp/model_{}'.format(id)
        self._filename = save_path
        reset_default_graph()

    def add_layer(self, size, act_fn):
        '''Add a layer to the model

        Args:
            size (int): number of neurons in the layer
            act_fn (str): activation function of the layer:
                - 'relu', 'sigmoid', 'softmax' or 'linear'
        '''

        self._layers.append(Layer(size, act_fn))

    def connect_layers(self):
        '''Fully connects each layer added using add_layer()
        '''

        for layer in range(len(self._layers) - 1):
            self.__weights.append(
                Variable(random_normal([
                    self._layers[layer].size,
                    self._layers[layer + 1].size
                ]), name='W_fc{}'.format(layer + 1))
            )
        for layer in range(1, len(self._layers)):
            self.__biases.append(
                Variable(random_normal([
                    self._layers[layer].size
                ]), name='B_fc{}'.format(layer))
            )

    def fit(self, x_l, y_l, learning_rate=0.1, train_epochs=500,
            keep_prob=1.0):
        '''Fits the neural network model using a set number of training iterations

        Args:
            x_l (numpy array): training inputs
            y_l (numpy array): training outputs
            learning_rate (float): learning rate during training
            train_epochs (int): number of training iterations
            keep_prob (float): probability that a neuron is not subject to
                dropout
        '''

        assert type(learning_rate) is float, \
            'Invalid learning_rate type: {}'.format(type(learning_rate))
        assert type(train_epochs) is int, \
            'Invalid train_epochs type: {}'.format(type(train_epochs))
        assert 0.0 <= keep_prob <= 1.0, \
            'Invalid keep_prob value: {}'.format(keep_prob)

        if len(y_l) is 0 or len(x_l) is 0:
            raise ValueError('Learning set cannot be empty - check data split')

        x = placeholder('float', [None, self._layers[0].size])
        y = placeholder('float', [None, self._layers[-1].size])

        pred = self.__feed_forward(x, keep_prob=keep_prob)
        cost = square(y - pred)
        optimizer = train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(cost)

        with Session() as sess:
            sess.run(global_variables_initializer())
            for _ in range(train_epochs):
                sess.run(optimizer, feed_dict={x: x_l, y: y_l})
            saver = train.Saver()
            saver.save(sess, self._filename)
        sess.close()

    def fit_validation(self, x_l, y_l, x_v, y_v, learning_rate=0.1,
                       max_epochs=10000, keep_prob=0.0):
        '''Fits the neural network model using periodic (every 250 epochs)
        validation; if validation set performance does not improve, training
        stops

        Args:
            x_l (numpy array): training inputs
            y_l (numpy array): training outputs
            x_v (numpy array): validation inputs
            y_v (numpy array): validation outputs
            learning_rate (float): learning rate during training
            max_epochs (int): maximum number of training iterations
            keep_prob (float): probability that a neuron is not subject to
                dropout
        '''

        assert type(learning_rate) is float, \
            'Invalid learning_rate type: {}'.format(type(learning_rate))
        assert type(max_epochs) is int, \
            'Invalid train_epochs type: {}'.format(type(train_epochs))
        assert 0.0 <= keep_prob <= 1.0, \
            'Invalid keep_prob value: {}'.format(keep_prob)

        if len(y_l) is 0 or len(x_l) is 0:
            raise ValueError(
                'Learning set cannot be empty - check data split'
            )
        if len(y_v) is 0 or len(x_v) is 0:
            raise ValueError(
                'Validation set cannot be empty - check data split'
            )

        x = placeholder('float', [None, self._layers[0].size])
        y = placeholder('float', [None, self._layers[-1].size])

        pred = self.__feed_forward(x, keep_prob=keep_prob)
        cost = square(y - pred)
        optimizer = train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(cost)

        with Session() as sess:
            sess.run(global_variables_initializer())

            valid_rmse_lowest = 100
            current_epoch = 0

            while current_epoch < max_epochs:
                sess.run(optimizer, feed_dict={x: x_l, y: y_l})
                current_epoch += 1
                if current_epoch % 250 == 0:
                    valid_rmse = self.__calc_rmse(
                        sess.run(pred, feed_dict={x: x_v}), y_v
                    )
                    if valid_rmse < valid_rmse_lowest:
                        valid_rmse_lowest = valid_rmse
                    elif valid_rmse > valid_rmse_lowest + (
                        0.05 * valid_rmse_lowest
                    ):
                        break

            saver = train.Saver()
            saver.save(sess, self._filename)
        sess.close()

    def use(self, x):
        '''Use the neural network on supplied input data

        Args:
            x (numpy array): supplied input data

        Returns:
            numpy array: predicted value for input data
        '''

        with Session() as sess:
            saver = train.Saver()
            saver.restore(sess, self._filename)
            results = self.__feed_forward(x).eval()
        sess.close()
        return results

    def save(self, filepath=None):
        '''Save the neural network for later use; results in a .sess (tensorflow)
        file and a .struct (neural network architecture) file

        Args:
            filepath (str): location to save the model
        '''

        assert type(filepath) is (str or None), \
            'Invalid filepath type: {}'.format(filepath)
        if filepath is None:
            filepath = self._filename
        with open('{}.struct'.format(filepath), 'wb') as arch_file:
            dump(self._layers, arch_file)
        self.__save_restore(self._filename, filepath)

    def load(self, filepath, use_arch_file=True):
        '''Load a previously saved neural network

        Args:
            filepath (str): location of the saved model
        '''

        assert type(filepath) is str, \
            'Invalid filepath type: {}'.format(filepath)
        self._filename = filepath
        if use_arch_file:
            with open('{}.struct'.format(self._filename), 'rb') as arch_file:
                self._layers = load(arch_file)
                self.connect_layers()
        self.__save_restore(filepath, self._filename)

    def __feed_forward(self, x, keep_prob=1.0):
        '''Private method: feeds data through the neural network, returns output
        of final layer

        Args:
            x (1d numpy array): input data to feed through the network
            keep_prob (float): probability that a neuron is retained (not
                subjected to dropout)

        Returns:
            1d numpy array: result of the final layer pass-through
        '''

        output = x
        for idx, layer in enumerate(self._layers[1:]):
            output = nn.dropout(layer.act_fn(
                add(matmul(
                    output,
                    self.__weights[idx]
                ), self.__biases[idx])
            ), keep_prob)
        return output

    @staticmethod
    def __save_restore(old, new):
        '''Private, static method: resaves model file

        Args:
            old (str): path to old file
            new (str): path to new file
        '''

        with Session() as sess:
            saver = train.Saver()
            saver.restore(sess, old)
            saver.save(sess, new)
        sess.close()

    def __calc_rmse(self, y_hat, y):
        '''Private method: calculates the RMSE of the validation set during
        validation training

        Args:
            y_hat (numpy array): predicted data
            y (numpy array): known data

        Returns:
            float: root mean squared error of predicted data
        '''

        try:
            return(nsqrt(((y_hat - y)**2).mean()))
        except:
            return(nsqrt(((asarray(y_hat) - asarray(y))**2).mean()))


def train_model(validate, sets, vars, save_path=None, id=None):
    '''Starts a process to train a neural network

    Args:
        validate (bool): whether to use periodic validation or not
        sets (PackagedData): object housing learning, validation, test data
        vars (dict): Server model configuration variables
        save_path (str): path to where model is saved
        id (int): id for the model, optional
    '''

    model = MultilayerPerceptron(
        save_path=save_path
    )
    model.add_layer(len(sets.learn_x[0]), vars['input_activation'])
    for layer in vars['hidden_layers']:
        model.add_layer(layer[0], layer[1])
    model.add_layer(len(sets.learn_y[0]), vars['output_activation'])
    model.connect_layers()
    if validate:
        model.fit_validation(
            sets.learn_x,
            sets.learn_y,
            sets.valid_x,
            sets.valid_y,
            learning_rate=vars['learning_rate'],
            max_epochs=vars['validation_max_epochs'],
            keep_prob=vars['keep_prob']
        )
    else:
        model.fit(
            sets.learn_x,
            sets.learn_y,
            learning_rate=vars['learning_rate'],
            train_epochs=vars['train_epochs'],
            keep_prob=vars['keep_prob']
        )
