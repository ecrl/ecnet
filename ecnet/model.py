#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet_model.py
#  
#  Developed in 2017 by Travis Kessler <Travis_Kessler@student.uml.edu>
#  
#  This program contains functions necessary creating, training, saving, and importing neural network models
#

import tensorflow as tf
import numpy as np
import pickle
import os
from functools import reduce

class multilayer_perceptron:
	# initialization of model structure
	def __init__(self):
		self.layers = []
		self.weights = []
		self.biases = []
		tf.reset_default_graph()
	
	# adds a skeleton for the layer: [number of neurons, activation function]
	def addLayer(self, num_neurons, function = "relu"):
		self.layers.append([num_neurons, function])
		
	# connects skeleton layers: results in random weights and biases
	def connectLayers(self):
		# weights
		for layer in range(0, len(self.layers)-1):
			self.weights.append(tf.Variable(tf.random_normal([self.layers[layer][0], self.layers[layer+1][0]]), name = "W_fc%d"%(layer + 1)))
		# biases
		for layer in range(1, len(self.layers)):
			self.biases.append(tf.Variable(tf.random_normal([self.layers[layer][0]]), name = "B_fc%d"%(layer)))

	# function for feeding data through the model, and returns the output
	def feed_forward(self, x):
		layerOutput = [x]
		for layer in range(1, len(self.layers)):
			# relu
			if "relu" in self.layers[layer][1]:
				layerOutput.append(tf.nn.relu(tf.add(tf.matmul(layerOutput[-1], self.weights[layer - 1]), self.biases[layer - 1])))
			# sigmoid
			elif "sigmoid" in self.layers[layer][1]:
				layerOutput.append(tf.nn.sigmoid(tf.add(tf.matmul(layerOutput[-1], self.weights[layer - 1]), self.biases[layer - 1])))
			# linear
			elif "linear" in self.layers[layer][1]:
				layerOutput.append(tf.add(tf.matmul(layerOutput[-1], self.weights[layer - 1]), self.biases[layer - 1]))
			elif "softmax" in self.layers[layer][1]:
				layerOutput.append(tf.nn.softmax(tf.add(tf.matmul(layerOutput[-1], self.weights[layer - 1]), self.biases[layer - 1])))
		return(layerOutput[-1])
	
	### Data is served to the model, and fits the model to the data
	def fit(self, x_l, y_l, learning_rate = 0.1, train_epochs = 500):
		# placeholder variables for input and output matrices
		x = tf.placeholder("float", [None, self.layers[0][0]])
		y = tf.placeholder("float", [None, self.layers[-1][0]])
		pred = self.feed_forward(x)
		
		# cost function and optimizer - TODO: look into other optimizers besides Adam
		cost = tf.square(y - pred)
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
		
		# opens the tensorflow session for training
		with tf.Session() as self.sess:
			# initialize the pre-defined variables
			self.sess.run(tf.global_variables_initializer())
			# runs training loop for explicit number of epochs -> find in config.yaml
			for epoch in range(train_epochs):
				self.sess.run(optimizer, feed_dict = {x: x_l, y: y_l})
			# saves a temporary output file, variables (weights, biases) included
			saver = tf.train.Saver()
			saver.save(self.sess,"./tmp/_ckpt")
		self.sess.close()
	
	### Data is served to the model, and fits the model to the data using periodic validation
	def fit_validation(self, x_l, x_v, y_l, y_v, learning_rate = 0.1, mdrmse_stop = 0.1, mdrmse_memory = 50, max_epochs = 500):
		# placeholder variables for input and output matrices
		x = tf.placeholder("float", [None, self.layers[0][0]])
		y = tf.placeholder("float", [None, self.layers[-1][0]])
		pred = self.feed_forward(x)
		
		# variables and arrays for validation process
		mdRMSE = 1
		current_epoch = 0
		rmse_list = []
		delta_list = []
	
		# cost function and optimizer - TODO: look into other optimizers besides Adam
		cost = tf.square(y - pred)
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
			
		# opens the tensorflow session for training
		with tf.Session() as self.sess:
			# initialize the pre-defined variables
			self.sess.run(tf.global_variables_initializer())
			# while current mdRMSE is more than the cutoff point, and the max num of epochs hasn't been reached:
			while mdRMSE > mdrmse_stop and current_epoch < max_epochs:
				self.sess.run(optimizer, feed_dict = {x: x_l, y: y_l})
				current_epoch += 1
				# determine new mdRMSE after every 100 epochs
				if current_epoch % 100 == 0:
					valid_pred = self.sess.run(pred, feed_dict = {x: x_v})
					rmse_list.append(calc_valid_rmse(valid_pred, y_v))
					if len(rmse_list) > 1:
						delta_list.append(abs(rmse_list[-2] - rmse_list[-1]))
						# mdRMSE memory: how far back the function looks to determine mdRMSE
						if len(delta_list) > mdrmse_memory:
							del(delta_list[0])
						mdRMSE = reduce(lambda x, y: x + y, delta_list) / len(delta_list)
			
			saver = tf.train.Saver()
			saver.save(self.sess, "./tmp/_ckpt")
		self.sess.close()
		
	### Tests the test data from the server
	def test_new(self, x):
		with tf.Session() as self.sess:
			saver = tf.train.Saver()
			saver.restore(self.sess, "./tmp/_ckpt")
			result = self.feed_forward(x)
			result = result.eval()
		self.sess.close()
		return result
		
	### Saves the _ckpt.ecnet file to a pre-defined output file	
	def save_net(self, output_filepath):
		with tf.Session() as self.sess:
			saver = tf.train.Saver()
			saver.restore(self.sess, "./tmp/_ckpt")
			saver.save(self.sess, "./" + output_filepath + ".sess")
		self.sess.close()
		pickle.dump(self.layers, open("./" + output_filepath + ".struct", "wb"))
		
	### Loads a pre-defined file into the model
	def load_net(self, model_load_filename):
		self.layers = pickle.load(open("./" + model_load_filename + ".struct", "rb"))
		self.connectLayers()
		with tf.Session() as self.sess:
			saver = tf.train.Saver()
			saver.restore(self.sess, "./" + model_load_filename + ".sess")
			saver.save(self.sess, "./tmp/_ckpt")
		self.sess.close()
		
	### Return numerical values for weights
	def export_weights(self):
		weights = []
		with tf.Session() as self.sess:
			saver = tf.train.Saver()
			saver.restore(self.sess, "./tmp/_ckpt")
			for i in range(0,len(self.weights)):
				weights.append(self.weights[i].eval())
		return weights
		
	### Return numerical values for biases
	def export_biases(self):
		biases = []
		with tf.Session() as self.sess:
			saver = tf.train.Saver()
			saver.restore(self.sess, "./tmp/_ckpt")
			for i in range(0,len(self.biases)):
				biases.append(self.biases[i].eval())
		return biases
		
def calc_valid_rmse(x, y):
	try:
		return(np.sqrt(((x-y)**2).mean()))
	except:
		try:
			return(np.sqrt(((np.asarray(x)-np.asarray(y))**2).mean()))
		except:
			print("Error in calculating RMSE. Check input data format.")
			sys.exit()
		
		
		
		
		
		
		
		
