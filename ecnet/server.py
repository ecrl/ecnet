#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet/server.py
#  v.1.4.3
#  Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
#  This file contains the "Server" class, which handles ECNet project creation,
#	neural network model creation, data hand-off to models, and project error
#	calculation. For example scripts, refer to https://github.com/tjkessler/ecnet
#

# 3rd party packages (open src.)
import yaml
import warnings
import os
import numpy as np
import zipfile
import pickle
from ecabc.abc import ABC

# ECNet source files
import ecnet.data_utils
import ecnet.error_utils
import ecnet.model
import ecnet.limit_parameters

'''
Server object: handles project creation/usage for ECNet projects, handles
data hand-off to neural networks for model training
'''
class Server:

	'''
	Initialization: imports configuration variables from *filename*
	'''
	def __init__(self, filename = 'config.yml'):

		# Dictionary containing configuration variables
		self.vars = {}

		# Open configuration file found at *filename*
		try:
			file = open(filename, 'r')
			self.vars.update(yaml.load(file))

		# Configuration file not found, create default 'config.yml'
		except:
			warnings.warn('WARNING: supplied configuration file not found: creating default config.yml')
			config_dict = {
				'data_filename' : 'data.csv',
				'data_sort_type' : 'random',
				'data_split' : [0.65,0.25,0.10],
				'learning_rate' : 0.1,
				'mlp_hidden_layers' : [[5, 'relu'], [5, 'relu']],
				'mlp_in_layer_activ' : 'relu',
				'mlp_out_layer_activ' : 'linear',
				'project_name' : 'my_project',
				'project_num_builds' : 1,
				'project_num_nodes' : 1,
				'project_num_trials' : 1,
				'project_print_feedback': True,
				'train_epochs' : 500,
				'valid_max_epochs': 5000
			}
			filename = 'config.yml'
			file = open(filename, 'w')
			yaml.dump(config_dict, file)
			self.vars.update(config_dict)

		# Set configuration filename
		self.config_filename = filename
		# Initial state of Server is to create single models
		self.using_project = False

	
	'''
	Creates the folder structure for a project; if not called, Server will create
	just one neural network model. A project consists of builds, each build
	containing nodes (node predictions are averaged for final build prediction).
	Number of builds and nodes are specified in the configuration file.
	'''
	def create_project(self, project_name = None):

		# If alternate project name is not given, use configuration file's project name
		if project_name is None:
			project_name = self.vars['project_name']
		# If alternate name is given, update Server config with new name
		else:
			self.vars['project_name'] = project_name
		# Create base project folder in working directory (if not already there)
		if not os.path.exists(project_name):
			os.makedirs(project_name)
		# For each build (number of builds specified in config):
		for build in range(self.vars['project_num_builds']):
			# Create build path (project + build number)
			path_b = os.path.join(project_name, 'build_%d' % build)
			# Create folder (if it doesn't exist)
			if not os.path.exists(path_b):
				os.makedirs(path_b)
			# For each node (number of nodes specified in config):
			for node in range(self.vars['project_num_nodes']):
				# Create node path (project + build number + node number)
				path_n = os.path.join(path_b, 'node_%d' % node)
				# Create folder (if it doesn't exits)
				if not os.path.exists(path_n):
					os.makedirs(path_n)
		# Update Server boolean, indicating we are now using a project (instead of single model)
		self.using_project = True

	'''
	Imports data from *data_filename*; utilizes 'data_utils' for file I/O, data set splitting
	and data set packaging for hand-off to neural network models
	'''
	def import_data(self, data_filename = None):

		# If no filename specified, use configuration filename
		if data_filename is None:
			data_filename = self.vars['data_filename']
		# Else, set config variable to supplied filename
		else:
			self.vars['data_filename'] = data_filename
		# Import the data using *data_utils* *DataFrame* object
		self.DataFrame = ecnet.data_utils.DataFrame(data_filename)
		# Create learning, validation and testing sets
		if self.vars['data_sort_type'] == 'random':
			self.DataFrame.create_sets(random = True, split = self.vars['data_split'])
		else:
			self.DataFrame.create_sets(random = False)
		# Package sets for model hand-off
		self.packaged_data = self.DataFrame.package_sets()

	'''
	Limits the input dimensionality of the currently loaded DataFrame to a dimension of *limit_num*.
	Saves the resulting limited DataFrame to *output_filename*. Option to *shuffle* data sets between
	inclusions/after each generation if using genetic algorithm. *use_genetic* allows for using a 
	genetic algorithm to limit the dimensionality (default to iterative inclusion), with arguments 
	for genetic algorithm *population_size*, *num_survivors* of each generation, and the number of 
	generations *num_generations* (PyGenetics package).
	'''
	def limit_parameters(self, limit_num, output_filename, use_genetic = False, population_size = 500, num_survivors = 200, num_generations = 25, shuffle = False):

		if use_genetic:
			params = ecnet.limit_parameters.limit_genetic(self.DataFrame, limit_num, population_size, num_survivors, num_generations, shuffle = shuffle, print_feedback = self.vars['project_print_feedback'])
		else:
			params = ecnet.limit_parameters.limit_iterative_include(self.DataFrame, limit_num)
		ecnet.limit_parameters.output(self.DataFrame, params, output_filename)

	'''
	Tunes the neural network learning hyperparameters (learning_rate, valid_max_epochs, neuron
	counts in each hidden layer) using an artificial bee colony algorithm (ecabc package)
	'''
	def tune_hyperparameters(self, target_score = None, iteration_amt = 50, amt_employers = 50):

		# Make sure project is not constructed
		if self.using_project:
			warnings.warn('WARNING: tune_hyperparameters() uses individual neural networks, not projects. Setting using_project boolean to false.')
			self.using_project = False

		'''
		Fitness function to be used by the artificial bee colony
		'''
		def test_neural_network(values):

			self.vars['learning_rate'] = values[0]
			self.vars['valid_max_epochs'] = values[1]
			self.vars['mlp_hidden_layers'][0][0] = values[2]
			self.vars['mlp_hidden_layers'][1][0] = values[3]

			self.train_model(validate = True)
			return self.calc_error('rmse', dset = 'test')['rmse']

		# Minimum and maximum values for hyperparameters (learning rate, valid_max_epochs, hidden layer neuron count)
		hyperparameters = [('float', (0.01, 0.2)), ('int', (1000, 25000)), ('int', (8, 32)), ('int', (8, 32))]

		# If *target_score* (RMSE) is not given, run ABC for *iteration_amt* iterations
		if target_score is None:
			abc = ABC(iterationAmount = iteration_amt, 
				fitnessFunction = test_neural_network, 
				valueRanges = hyperparameters, 
				amountOfEmployers = amt_employers)

		# Else, run ABC until *target_score* is reached
		else:
			abc = ABC(endValue = target_score,
				fitnessFunction = test_neural_network,
				valueRanges = hyperparameters,
				amountOfEmployers = amt_employers)

		# Run the artificial bee colony
		abc.printInfo(self.vars['project_print_feedback'])
		new_hyperparameters = abc.runABC()

		# Set Server hyperparameters to ABC-calculated hyperparameters
		self.vars['learning_rate'] = new_hyperparameters[0]
		self.vars['valid_max_epochs'] = new_hyperparameters[1]
		self.vars['mlp_hidden_layers'][0][0] = new_hyperparameters[2]
		self.vars['mlp_hidden_layers'][1][0] = new_hyperparameters[3]

		# Return ABC-calculated hyperparameters
		return new_hyperparameters

	'''
	Trains a neural network (multilayer perceptron) using learning data, and validation data (if *validate*
	== True). *args is used to specify shuffling of data sets for each trial; use "shuffle_lv" (shuffles 
	training data) or "shuffle_lvt" (shuffles all data)
	'''
	def train_model(self, *args, validate = False):

		# Not using project, train single model
		if not self.using_project:
			# Create the model, train the model, save the model to temp folder
			mlp_model = self.__create_mlp_model()
			# Use validation sets to periodically test model's performance and determine when to stop;
			#	prevents overfitting
			if validate:
				mlp_model.fit_validation(
					self.packaged_data.learn_x,
					self.packaged_data.learn_y,
					self.packaged_data.valid_x,
					self.packaged_data.valid_y,
					self.vars['learning_rate'],
					self.vars['valid_max_epochs'])
			# No validation is used, just train for 'training_epochs' iterations
			else:
				mlp_model.fit(
					self.packaged_data.learn_x,
					self.packaged_data.learn_y,
					self.vars['learning_rate'],
					self.vars['train_epochs'])
			mlp_model.save('./tmp/model_output')

		# Project is constructed, create models according to configuration specifications
		else:
			# For each build:
			for build in range(self.vars['project_num_builds']):
				# For each node:
				for node in range(self.vars['project_num_nodes']):
					# For each trial:
					for trial in range(self.vars['project_num_trials']):
						# Print status update (if config variable is True)
						if self.vars['project_print_feedback']:
							print('Build %d, Node %d, Trial %d...' % (build + 1, node + 1, trial + 1))
						# Determine filepath where trial will be saved
						path_b = os.path.join(self.vars['project_name'], 'build_%d' % build)
						path_n = os.path.join(path_b, 'node_%d' % node)
						path_t = os.path.join(path_n, 'trial_%d' % trial)
						# Create the model, train the model, save the model to trial filepath
						mlp_model = self.__create_mlp_model()
						# Use validation sets to periodically test model's performance and determine when done training
						if validate:
							mlp_model.fit_validation(
								self.packaged_data.learn_x,
								self.packaged_data.learn_y,
								self.packaged_data.valid_x,
								self.packaged_data.valid_y,
								self.vars['learning_rate'],
								self.vars['valid_max_epochs'])
						# No validation is used, just train for 'training_epochs' iterations
						else:
							mlp_model.fit(
								self.packaged_data.learn_x,
								self.packaged_data.learn_y,
								self.vars['learning_rate'],
								self.vars['train_epochs'])
						mlp_model.save(path_t)
						# Shuffle the training data sets
						if 'shuffle_lv' in args:
							self.DataFrame.shuffle('l', 'v', split = self.vars['data_split'])
							self.packaged_data = self.DataFrame.package_sets()
						# Shuffle all data sets
						elif 'shuffle_lvt' in args:
							self.DataFrame.create_sets(split = self.vars['data_split'])
							self.packaged_data = self.DataFrame.package_sets()

	'''
	Selects the best performing model from each node for each build to represent
	the node (build prediction = average of node predictions). Selection of best
	model is based on data set *dset* performance. This method may take a while,
	depending on project size.
	'''
	def select_best(self, dset = None, error_fn = 'rmse'):

		# If not using a project, no need to call this function!
		if not self.using_project:
			raise Exception('ERROR: Project is not created; project structure is required to select best models!')
		# Using a project
		else:
			# Print status update (if config variable is True)
			print('Selecting best models from each node for each build...')
			# Determine input values and target values to use for selection (based on dset arg)
			x_vals = self.__determine_x_vals(dset)
			y_vals = self.__determine_y_vals(dset)
			# For each build:
			for build in range(self.vars['project_num_builds']):
				# Determine build path
				path_b = os.path.join(self.vars['project_name'], 'build_%d' % build)
				# For each node:
				for node in range(self.vars['project_num_nodes']):
					# Determine node path
					path_n = os.path.join(path_b, 'node_%d' % node)
					# List of trial errors within the node
					rmse_list = []
					# For each trial:
					for trial in range(self.vars['project_num_trials']):
						# Create model, load trial, calculate error, append to list
						mlp_model = ecnet.model.MultilayerPerceptron()
						mlp_model.load(os.path.join(path_n, 'trial_%d' % trial))
						rmse_list.append(self.__error_fn(error_fn, mlp_model.use(x_vals), y_vals))
					# Determines the lowest error in error list
					current_min = 0
					for new_min in range(len(rmse_list)):
						if rmse_list[new_min] < rmse_list[current_min]:
							current_min = new_min
					# Load the model with the lowest error, resave as 'final_net' in the node folder
					mlp_model = ecnet.model.MultilayerPerceptron()
					mlp_model.load(os.path.join(path_n, 'trial_%d' % current_min))
					mlp_model.save(os.path.join(path_n, 'final_net'))

	'''
	Use trained neural network (multilayer perceptron), either single or build,
	to predict values for specified *dset*
	'''
	def use_model(self, dset = None):

		# Determine data set to be passed to model, specified by *dset*
		x_vals = self.__determine_x_vals(dset)
		# Not using project, use single model
		if not self.using_project:
			# Create model object
			mlp_model = ecnet.model.MultilayerPerceptron()
			# Load the trained model
			mlp_model.load('./tmp/model_output')
			# Return results obtained from model
			return [mlp_model.use(x_vals)]
		# Project is constructed, use project builds to predict values
		else:
			# List of final predictions
			preds = []
			# For each project build:
			for build in range(self.vars['project_num_builds']):
				# Determine build path
				path_b = os.path.join(self.vars['project_name'], 'build_%d' % build)
				# Build predictions (one from each node)
				build_preds = []
				# For each node:
				for node in range(self.vars['project_num_nodes']):
					# Determine node path
					path_n = os.path.join(path_b, 'node_%d' % node)
					# Determine final build (from select_best) path
					path_f = os.path.join(path_n, 'final_net')
					# Create model, load net, append results
					mlp_model = ecnet.model.MultilayerPerceptron()
					mlp_model.load(path_f)
					build_preds.append(mlp_model.use(x_vals))
				# Average build prediction = average of node predictions
				ave_build_preds = []
				# For each data point
				for point in range(len(build_preds[0])):
					# List of node predictions for individual data point
					local_pred = []
					# For each node prediction
					for pred in range(len(build_preds)):
						# Append node prediction for data point
						local_pred.append(build_preds[pred][point])
					# Compute average of node predictions for point, append to ave list
					ave_build_preds.append(sum(local_pred) / len(local_pred))
				# Append average build prediction to list of final predictions
				preds.append(list(ave_build_preds))
			# Return final predictions
			return preds

	'''
	Calculates and returns errors based on input *args; possible arguments are *rmse*,
	*r2* (r-squared correlation coefficient), *mean_abs_error*, *med_abs_error*. Multiple
	error arguments can be supplied. *dset* argument specifies which data set the error 
	is being calculated for (e.g. 'test', 'train'). Returns dictionary of error values.
	'''
	def calc_error(self, *args, dset = None):
		# Dictionary keys = error arguments, values = error values
		error_dict = {}
		# Obtain predictions for specified data set
		y_hat = self.use_model(dset)
		# Determine target values for specified data set
		y = self.__determine_y_vals(dset)
		# For each supplied error argument:
		for arg in args:
			# Using project
			if self.using_project:
				# List of error for each build
				error_list = []
				# For each build's prediction:
				for pred in y_hat:
					# Append build error to error list
					error_list.append(self.__error_fn(arg, pred, y))
				# Key error = error list
				error_dict[arg] = error_list
			# Using single model
			else:
				# Key error = calculated error
				error_dict[arg] = self.__error_fn(arg, y_hat, y)
		# Return error dictionary
		return error_dict

	'''
	Outputs the *results* obtained from "use_model()" to a specified *filename*
	'''
	def output_results(self, results, filename = 'my_results.csv'):

		# Output results using data_utils function
		ecnet.data_utils.output_results(results, self.DataFrame, filename)

	'''
	Saves the current state of Server (including currently imported DataFrame and configuration), 
	cleans up the project directory if specified in *clean_up* (only keeps final node models),
	and zips up the current state and project directory into a .project file
	'''
	def save_project(self, clean_up = True):

		# If removing trials from project directory (keeping final models):
		if clean_up:
			# For each build
			for build in range(self.vars['project_num_builds']):
				path_b = os.path.join(self.vars['project_name'], 'build_%d' % build)
				# For each node
				for node in range(self.vars['project_num_nodes']):
					path_n = os.path.join(path_b, 'node_%d' % node)
					# Remove trials
					trial_files = [file for file in os.listdir(path_n) if 'trial' in file]
					for file in trial_files:
						os.remove(os.path.join(path_n, file))

		# Save Server configuration to configuration YAML file
		with open(self.config_filename, 'w') as config_file:
			yaml.dump(self.vars, config_file, default_flow_style = False, explicit_start = True)
		config_file.close()

		# Save currently loaded DataFrame
		with open(os.path.join(self.vars['project_name'], 'data.d'), 'wb') as data_file:
			pickle.dump(self.DataFrame, data_file)
		data_file.close()

		# Zip up all files in project directory, save to .project file
		zip_file = zipfile.ZipFile(self.vars['project_name'] + '.project', 'w', zipfile.ZIP_DEFLATED)
		for root, dirs, files in os.walk(self.vars['project_name']):
			for file in files:
				zip_file.write(os.path.join(root, file))
		zip_file.close()

	'''
	Opens a .project file, imports configuration and last used data set, unzips model files
	to project directory
	'''
	def open_project(self, filename):

		# Check for .project file format
		if '.project' not in filename:
			filename += '.project'

		# Extract project directory from .project file
		zip_file = zipfile.ZipFile(filename, 'r')
		zip_file.extractall('./')
		zip_file.close()

		# Import project configuration
		with open(self.config_filename, 'r') as config_file:
			self.vars.update(yaml.load(config_file))
		config_file.close()

		# Import last used DataFrame
		with open(os.path.join(self.vars['project_name'], 'data.d'), 'rb') as data_file:
			self.DataFrame = pickle.load(data_file)
		data_file.close()

		# Package data for model usage
		self.packaged_data = self.DataFrame.package_sets()

		# Set project use boolean to True
		self.using_project = True

	'''
	PRIVATE METHOD: Helper function for determining data set *dset* to be passed to the model (inputs)
	'''
	def __determine_x_vals(self, dset):

		# Use the test set
		if dset == 'test':
			return self.packaged_data.test_x
		# Use the validation set
		elif dset == 'valid':
			return self.packaged_data.valid_x
		# Use the learning set
		elif dset == 'learn':
			return self.packaged_data.learn_x
		# Use training set (learning and validation)
		elif dset == 'train':
			x_vals = []
			for val in self.packaged_data.learn_x:
				x_vals.append(val)
			for val in self.packaged_data.valid_x:
				x_vals.append(val)
			return np.asarray(x_vals)
		# Use all data sets (learning, validation and testing)
		else:
			x_vals = []
			for val in self.packaged_data.learn_x:
				x_vals.append(val)
			for val in self.packaged_data.valid_x:
				x_vals.append(val)
			for val in self.packaged_data.test_x:
				x_vals.append(val)
			return np.asarray(x_vals)

	'''
	PRIVATE METHOD: Helper function for determining data set *dset* to be passed to the model (targets)
	'''
	def __determine_y_vals(self, dset):

		# Use the test set
		if dset == 'test':
			return self.packaged_data.test_y
		# Use the validation set
		elif dset == 'valid':
			return self.packaged_data.valid_y
		# Use the learning set
		elif dset == 'learn':
			return self.packaged_data.learn_y
		# Use training set (learning and validation)
		elif dset == 'train':
			y_vals = []
			for val in self.packaged_data.learn_y:
				y_vals.append(val)
			for val in self.packaged_data.valid_y:
				y_vals.append(val)
			return np.asarray(y_vals)
		# Use all data sets (learning, validation and testing)
		else:
			y_vals = []
			for val in self.packaged_data.learn_y:
				y_vals.append(val)
			for val in self.packaged_data.valid_y:
				y_vals.append(val)
			for val in self.packaged_data.test_y:
				y_vals.append(val)
			return np.asarray(y_vals)

	'''
	PRIVATE METHOD: Helper function for creating a neural network (multilayer perceptron)
	'''
	def __create_mlp_model(self):

		# Create the model object
		mlp_model = ecnet.model.MultilayerPerceptron()
		# Add input layer, size = number of data inputs, activation function specified in configuration file
		mlp_model.add_layer(self.DataFrame.num_inputs, self.vars['mlp_in_layer_activ'])
		# Add hidden layers, sizes and activation functions specified in configuration file
		for hidden in range(len(self.vars['mlp_hidden_layers'])):
			mlp_model.add_layer(self.vars['mlp_hidden_layers'][hidden][0], self.vars['mlp_hidden_layers'][hidden][1])
		# Add output layer, size = number of data targets, activation function specified in configuration file
		mlp_model.add_layer(self.DataFrame.num_targets, self.vars['mlp_out_layer_activ'])
		# Connect layers (compute initial weights and biases)
		mlp_model.connect_layers()
		# Return the model object
		return mlp_model

	'''
	PRIVATE METHOD: used to parse error argument, calculate specified error and return it
	'''
	def __error_fn(self, arg, y_hat, y):
		if arg == 'rmse':
			return ecnet.error_utils.calc_rmse(y_hat, y)
		elif arg == 'r2':
			return ecnet.error_utils.calc_r2(y_hat, y)
		elif arg == 'mean_abs_error':
			return ecnet.error_utils.calc_mean_abs_error(y_hat, y)
		elif arg == 'med_abs_error':
			return ecnet.error_utils.calc_med_abs_error(y_hat, y)
		else:
			raise Exception('ERROR: Unknown/unsupported error function')
