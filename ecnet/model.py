#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/error_utils.py
# v.1.7.0
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

    def __init__(self, id=0):
        '''Neural network (multilayer perceptron) - contains methods for training,
        using, saving and opening neural network models
        '''

        self.__layers = []
        self.__weights = []
        self.__biases = []
        self.__id = id
        if not path.isdir('./tmp/'):
            mkdir('./tmp/')
        reset_default_graph()

    def add_layer(self, size, act_fn):
        '''
        Add a layer to the model

        Args:
            size (int): number of neurons in the layer
            act_fn (str): activation function of the layer:
                - 'relu', 'sigmoid', 'softmax' or 'linear'
        '''

        self.__layers.append(Layer(size, act_fn))

    def connect_layers(self):
        '''Fully connects each layer added using add_layer()
        '''

        for layer in range(len(self.__layers) - 1):
            self.__weights.append(
                Variable(random_normal([
                    self.__layers[layer].size,
                    self.__layers[layer + 1].size
                ]), name='W_fc{}'.format(layer + 1))
            )
        for layer in range(1, len(self.__layers)):
            self.__biases.append(
                Variable(random_normal([
                    self.__layers[layer].size
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
            keep_prob (float): probability that a neuron is retained (not
                subjected to dropout)
        '''

        if len(y_l) is 0 or len(x_l) is 0:
            raise ValueError('Learning set cannot be empty - check data split')

        x = placeholder('float', [None, self.__layers[0].size])
        y = placeholder('float', [None, self.__layers[-1].size])

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
            saver.save(sess, './tmp/_m{}/_ckpt'.format(self.__id))
        sess.close()

    def fit_validation(self, x_l, y_l, x_v, y_v, learning_rate=0.1,
                       max_epochs=1500, keep_prob=1.0):
        '''Fits the neural network model using periodic (every 250 epochs)
        validation; if validation set performance worsens, training is complete

        Args:
            x_l (numpy array): training inputs
            y_l (numpy array): training outputs
            x_v (numpy array): validation inputs
            y_v (numpy array): validation outputs
            learning_rate (float): learning rate during training
            max_epochs (int): maximum number of training iterations
            keep_prob (float): probability that a neuron is retained (not
                subjected to dropout)
        '''

        if len(y_l) is 0 or len(x_l) is 0:
            raise ValueError(
                'Learning set cannot be empty - check data split'
            )
        if len(y_v) is 0 or len(x_v) is 0:
            raise ValueError(
                'Validation set cannot be empty - check data split'
            )

        x = placeholder('float', [None, self.__layers[0].size])
        y = placeholder('float', [None, self.__layers[-1].size])

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
            saver.save(sess, './tmp/_m{}/_ckpt'.format(self.__id))
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
            saver.restore(sess, './tmp/_m{}/_ckpt'.format(self.__id))
            results = self.__feed_forward(x).eval()
        sess.close()
        return results

    def save(self, filepath):
        '''Save the neural network for later use; results in a .sess (tensorflow)
        file and a .struct (neural network architecture) file

        Args:
            filepath (str): location to save the model
        '''

        with Session() as sess:
            saver = train.Saver()
            saver.restore(sess, './tmp/_m{}/_ckpt'.format(self.__id))
            saver.save(sess, filepath)
        sess.close()
        architecture_file = open('{}.struct'.format(filepath), 'wb')
        dump(self.__layers, architecture_file)
        architecture_file.close()

    def load(self, filepath):
        '''Load a previously saved neural network

        Args:
            filepath (str): location of the saved model
        '''

        architecture_file = open('{}.struct'.format(filepath), 'rb')
        self.__layers = load(architecture_file)
        self.connect_layers()
        with Session() as sess:
            saver = train.Saver()
            saver.restore(sess, filepath)
            saver.save(sess, './tmp/_m{}/_ckpt'.format(self.__id))
        sess.close()

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
        for idx, layer in enumerate(self.__layers[1:]):
            output = nn.dropout(layer.act_fn(
                add(matmul(
                    output,
                    self.__weights[idx]
                ), self.__biases[idx])
            ), keep_prob)
        return output

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
