#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet/error_utils.py
#  v.1.4.3.1
#  Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
#  This program contains functions necessary creating, training, saving, and reusing neural network models
#

# 3rd party packages (open src.)
import tensorflow as tf
import numpy as np
import pickle
from functools import reduce

'''
Basic neural network (multilayer perceptron); contains methods for adding layers
with specified neuron counts and activation functions, training the model, using
the model on new data, saving the model for later use, and reusing a previously
trained model
'''
class MultilayerPerceptron:

	'''
	Initialization of object
	'''
	def __init__(self):

		self.layers = []
		self.weights = []
		self.biases = []
		tf.reset_default_graph()

	'''
	Layer definition object, containing layer size and activation function to be used
	'''
	class Layer:

		def __init__(self, size, act_fn):

			self.size = size
			self.act_fn = act_fn

	'''
	Adds a layer definition to the model; default activation function is ReLU
	'''
	def add_layer(self, size, act_fn = 'relu'):

		self.layers.append(self.Layer(size, act_fn))

	'''
	Connects the layers in *self.layers* by creating weight matrices, bias vectors
	'''
	def connect_layers(self):

		# Create weight matrices (size = layer_n by layer_n+1)
		for layer in range(len(self.layers) - 1):
			self.weights.append(tf.Variable(tf.random_normal([self.layers[layer].size, self.layers[layer + 1].size]), name = 'W_fc%d' % (layer + 1)))
		# Create bias vectors (size = layer_n)
		for layer in range(1, len(self.layers)):
			self.biases.append(tf.Variable(tf.random_normal([self.layers[layer].size]), name = 'B_fc%d' % (layer)))

	'''
	Fits the neural network model using input data *x_l* and target data *y_l*. Optional arguments:
	*learning_rate* (training speed of the model) and *train_epochs* (number of traning iterations).
	'''
	def fit(self, x_l, y_l, learning_rate = 0.1, train_epochs = 500):

		# Check to see if learning set exists (set is not empty)
		if len(y_l) is 0 or len(x_l) is 0:
			raise ValueError('ERROR: Learning set cannot be empty! Check your data_split!')

		# TensorFlow placeholder variables for inputs and targets
		x = tf.placeholder('float', [None, self.layers[0].size])
		y = tf.placeholder('float', [None, self.layers[-1].size])

		# Predictions = *__feed_forward* final output
		pred = self.__feed_forward(x)

		# Cost function = squared error between targets and predictions
		cost = tf.square(y - pred)

		# Optimizer = AdamOptimizer (TODO: Look into other optimizers)
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

		# Run the TensorFlow session
		with tf.Session() as sess:
			# Initialize TensorFlow variables (weights, biases, placeholders)
			sess.run(tf.global_variables_initializer())
			# Train for *train_epochs* iterations
			for epoch in range(train_epochs):
				sess.run(optimizer, feed_dict = {x: x_l, y: y_l})
			# Saves model (weights and biases) to temporary output file
			saver = tf.train.Saver()
			saver.save(sess, './tmp/_ckpt')
		# Finish the TensorFlow session
		sess.close()

	'''
	Fits the neural network model using input data *x_l* and target data *y_l*, validating
	the learning process periodically based on validation data (*x_v* and *y_v*) performance).
	Optional arguments: *learning_rate* (training speed of the model), *max_epochs* (cutoff 
	point if training takes too long)
	'''
	def fit_validation(self, x_l, y_l, x_v, y_v, learning_rate = 0.1, max_epochs = 2500):

		# Check to see if learning and validation sets exists (sets are not empty)
		if len(y_l) is 0 or len(x_l) is 0:
			raise ValueError('ERROR: Learning set cannot be empty! Check your data_split!')
		if len(y_v) is 0 or len(x_v) is 0:
			raise ValueError('ERROR: Validation set cannot be empty! Check your data_split!')

		# TensorFlow placeholder variables for inputs and targets
		x = tf.placeholder('float', [None, self.layers[0].size])
		y = tf.placeholder('float', [None, self.layers[-1].size])

		# Predictions = *__feed_forward* final output
		pred = self.__feed_forward(x)

		# Cost function = squared error between targets and predictions
		cost = tf.square(y - pred)

		# Optimizer = AdamOptimizer (TODO: Look into other optimizers)
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

		# Run the TensorFlow session
		with tf.Session() as sess:
			# Initialize TensorFlow variables (weights, biases, placeholders)
			sess.run(tf.global_variables_initializer())
			
			# Lowest validation RMSE value (used as cutoff if validation RMSE rises above by 5%)
			valid_rmse_lowest = 50
			current_epoch = 0

			while current_epoch < max_epochs:
				# Run training iteration
				sess.run(optimizer, feed_dict = {x: x_l, y: y_l})
				# Current epoch ++
				current_epoch += 1
				# Every 250 epochs (TODO: make this a variable?):
				if current_epoch % 250 == 0:
					# Calculate validation set RMSE
					valid_rmse = self.__calc_rmse(sess.run(pred, feed_dict = {x: x_v}), y_v)
					# If RMSE is less than the current lowest RMSE, make new lowest RMSE
					if valid_rmse < valid_rmse_lowest:
						valid_rmse_lowest = valid_rmse
					# If RMSE is greater than the current lowest + 5%, done with training
					elif valid_rmse > valid_rmse_lowest + (0.05 * valid_rmse_lowest):
						break

			# Done training, save the model to temporary file
			saver = tf.train.Saver()
			saver.save(sess, './tmp/_ckpt')
		# Finish the TensorFlow session
		sess.close()

	'''
	Use the neural network model on input data *x*, returns *result*
	'''
	def use(self, x):

		# Run the TensorFlow session
		with tf.Session() as sess:
			# Import temporary output file containing weights and biases
			saver = tf.train.Saver()
			saver.restore(sess, './tmp/_ckpt')
			# Evaluate result
			result = self.__feed_forward(x).eval()
		# Finish the TensorFlow session
		sess.close()
		# Return the result
		return result

	'''
	Saves the neural network model (outside of temp file) to *filepath* for later use
	'''
	def save(self, filepath):

		# Run the TensorFlow session
		with tf.Session() as sess:
			# Import temporary output file containing weights and biases
			saver = tf.train.Saver()
			saver.restore(sess, './tmp/_ckpt')
			# Resave temporary file to file specified by *filepath*
			saver.save(sess, './' + filepath)
		# Finish the TensorFlow session
		sess.close()
		# Save the neural network model's architecture (layer sizes, activation functions)
		architecture_file = open('./' + filepath + '.struct', 'wb')
		pickle.dump(self.layers, architecture_file)
		architecture_file.close()

	'''
	Loads the neural network model found at *filepath*
	'''
	def load(self, filepath):

		# Import the architecture 'struct' file (layer sizes, activation functions)
		architecture_file = open('./' + filepath + '.struct', 'rb')
		self.layers = pickle.load(architecture_file)
		architecture_file.close()
		# Redefine weights and biases
		self.connect_layers()
		# Run the TensorFlow session
		with tf.Session() as sess:
			# Import file containing weights and biases
			saver = tf.train.Saver()
			saver.restore(sess, './' + filepath)
			# Save weights and biases to temporary file for use by 'fit', 'use', etc.
			saver.save(sess, './tmp/_ckpt')
		# Finish the TensorFlow session
		sess.close()

	'''
	PRIVATE METHOD: Feeds data through the neural network, returns output of final layer
	'''
	def __feed_forward(self, x):

		# First values to matrix multiply are the inputs
		output = x
		# For each layer (after the first layer, input)
		for index, layer in enumerate(self.layers[1:]):
			# ReLU activation function
			if layer.act_fn == 'relu':
				output = tf.nn.relu(tf.add(tf.matmul(output, self.weights[index]), self.biases[index]))
			# Sigmoid activation function
			elif layer.act_fn == 'sigmoid':
				output = tf.nn.relu(tf.add(tf.matmul(output, self.weights[index]), self.biases[index]))
			# Linear activation function
			elif layer.act_fn == 'linear':
				output = tf.add(tf.matmul(output, self.weights[index]), self.biases[index])
			# Softmax activation function
			elif layer.act_fn == 'softmax':
				output = tf.nn.softmax(tf.add(tf.matmul(output, self.weights[index]), self.biases[index]))

		# Return the final layer's output
		return output

	'''
	PRIVATE METHOD: Calculates the RMSE of the validation set during training
	'''
	def __calc_rmse(self, y_hat, y):
		try:
			return(np.sqrt(((y_hat-y)**2).mean()))
		except:
			try:
				return(np.sqrt(((np.asarray(y_hat)-np.asarray(y))**2).mean()))
			except:
				raise Exception('ERROR: Unable to calculate RMSE. Check input data format.')