#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet/limit_parameters.py
#  v.1.4.3.1
#  Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#  
#  This program contains the functions necessary for reducing the input dimensionality of a 
#	database to the most influential input parameters, using either an iterative inclusion
#	algorithm (limit_iterative_include) or a genetic algorithm (limit_genetic).
#

# 3rd party packages (open src.)
import csv
import copy
from pygenetics.ga_core import Population
from pygenetics.selection_functions import minimize_best_n

# ECNet source files
import ecnet.model
import ecnet.error_utils

'''
Limits the dimensionality of input data found in supplied *DataFrame* object to a
dimensionality of *limit_num* using iterative inclusion
'''
def limit_iterative_include(DataFrame, limit_num):

	# List of retained input parameters
	retained_input_list = []

	# Initialization of retained input lists
	learn_input_retained = []
	valid_input_retained = []
	test_input_retained = []

	# Obtain Numpy arrays for learning, validation, testing sets
	packaged_data = DataFrame.package_sets()

	# Until specified number of paramters *limit_num* are retained
	while len(retained_input_list) < limit_num:

		# List of RMSE's for currently retained inputs + new inputs to test
		retained_rmse_list = []

		# For all input paramters to test
		for idx, param in enumerate(DataFrame.input_names):

			# Obtain input parameter column for learning, validation and test sets
			learn_input_add = [[sublist[idx]] for sublist in packaged_data.learn_x]
			valid_input_add = [[sublist[idx]] for sublist in packaged_data.valid_x]
			test_input_add = [[sublist[idx]] for sublist in packaged_data.test_x]

			# No retained input parameters, inputs = individual parameters to be tested
			if len(retained_input_list) is 0:
				learn_input = learn_input_add
				valid_input = valid_input_add
				test_input = test_input_add
			# Else:
			else:
				# Inputs = currently retained input parameters
				learn_input = copy.deepcopy(learn_input_retained)
				valid_input = copy.deepcopy(valid_input_retained)
				test_input = copy.deepcopy(test_input_retained)
				# Add new input parameter to inputs
				for idx_add, param_add in enumerate(learn_input_add):
					learn_input[idx_add].append(param_add[0])
				for idx_add, param_add in enumerate(valid_input_add):
					valid_input[idx_add].append(param_add[0])
				for idx_add, param_add in enumerate(test_input_add):
					test_input[idx_add].append(param_add[0])

			# Create neural network model
			mlp_model = ecnet.model.MultilayerPerceptron()
			mlp_model.add_layer(len(learn_input[0]), 'relu')
			mlp_model.add_layer(8, 'relu')
			mlp_model.add_layer(8, 'relu')
			mlp_model.add_layer(len(packaged_data.learn_y[0]), 'linear')
			mlp_model.connect_layers()

			# Fit the model using validation
			mlp_model.fit_validation(
				learn_input,
				packaged_data.learn_y,
				valid_input,
				packaged_data.valid_y,
				max_epochs = 5000)

			# Calculate error for test set results, append to rmse list
			retained_rmse_list.append(ecnet.error_utils.calc_rmse(
				mlp_model.use(test_input),
				packaged_data.test_y))

		# Obtain index, value of best performing input paramter addition
		rmse_val, rmse_idx = min((rmse_val, rmse_idx) for (rmse_idx, rmse_val) in enumerate(retained_rmse_list))

		# Obtain input parameter addition with lowest error
		learn_retain_add = [[sublist[rmse_idx]] for sublist in packaged_data.learn_x]
		valid_retain_add = [[sublist[rmse_idx]] for sublist in packaged_data.valid_x]
		test_retain_add = [[sublist[rmse_idx]] for sublist in packaged_data.test_x]

		# No retained input parameters, retained = lowest error input parameter
		if len(retained_input_list) is 0:
			learn_input_retained = learn_retain_add
			valid_input_retained = valid_retain_add
			test_input_retained = test_retain_add
		# Else:
		else:
			# Append lowest error input parameter to retained parameters
			for idx, param in enumerate(learn_retain_add):
				learn_input_retained[idx].append(param[0])
			for idx, param in enumerate(valid_retain_add):
				valid_input_retained[idx].append(param[0])
			for idx, param in enumerate(test_retain_add):
				test_input_retained[idx].append(param[0])

		# Append name of retained input parameter to retained list
		retained_input_list.append(DataFrame.input_names[rmse_idx])
		# List currently retained input parameters
		print(retained_input_list)
		print(rmse_val)
		print()

	# Compiled *limit_num* input parameters, return list of retained parameters
	return retained_input_list

'''
Limits the dimensionality of input data found in supplied *DataFrame* object to a
dimensionality of *limit_num* using a genetic algorithm. Optional arguments for
*population_size* of genetic algorithm's population, *num_survivors* for selecting
the best performers from each population generation to reproduce, *num_generations*
for the number of times the population will reproduce, *shuffle* for shuffling the
data sets for each population member, and *print_feedback* for printing the average 
fitness score of the population after each generation.
'''
def limit_genetic(DataFrame, limit_num, population_size, num_survivors, num_generations, shuffle = False, print_feedback = True):

	'''
	Genetic algorithm cost function, supplied to the genetic algorithm; returns the RMSE
	of the test set results from a model constructed using the current permutation of
	input parameters *feed_dict* supplied by the genetic algorithm
	'''
	def ecnet_limit_inputs(feed_dict):

		# Set up learning, validation and testing sets
		learn_input = []
		valid_input = []
		test_input = []

		# If shuffling the data sets for each population member
		if shuffle:
			# Shuffle all sets
			DataFrame.shuffle('l', 'v', 't')
			# Obtain Numpy arrays for learning, validation, testing sets
			packaged_data_cf = DataFrame.package_sets()
		# Not shuffling, use sets defined on limit_genetic function call
		else:
			packaged_data_cf = packaged_data

		# For the input parameters chosen by the genetic algorithm:
		for param in feed_dict:

			# Grab the input parameter
			learn_input_add = [[sublist[feed_dict[param]]] for sublist in packaged_data_cf.learn_x]
			valid_input_add = [[sublist[feed_dict[param]]] for sublist in packaged_data_cf.valid_x]
			test_input_add = [[sublist[feed_dict[param]]] for sublist in packaged_data_cf.test_x]

			# Currently empty sets, sets = add lists
			if len(learn_input) == 0:
				learn_input = learn_input_add
				valid_input = valid_input_add
				test_input = test_input_add
			# Append add lists to sets
			else:
				for idx_add, param_add in enumerate(learn_input_add):
					learn_input[idx_add].append(param_add[0])
				for idx_add, param_add in enumerate(valid_input_add):
					valid_input[idx_add].append(param_add[0])
				for idx_add, param_add in enumerate(test_input_add):
					test_input[idx_add].append(param_add[0])
		
		# Construct a neural network (multilayer perceptron) model
		mlp_model = ecnet.model.MultilayerPerceptron()
		mlp_model.add_layer(len(learn_input[0]), 'relu')
		mlp_model.add_layer(8, 'relu')
		mlp_model.add_layer(8, 'relu')
		mlp_model.add_layer(len(packaged_data_cf.learn_y[0]), 'linear')
		mlp_model.connect_layers()

		# Train the model using validation
		mlp_model.fit_validation(
			learn_input,
			packaged_data_cf.learn_y,
			valid_input,
			packaged_data_cf.valid_y,
			max_epochs = 5000)

		# Returned fitness value = test set performance
		return ecnet.error_utils.calc_rmse(mlp_model.use(test_input), packaged_data_cf.test_y)

	# Package data for training/testing
	packaged_data = DataFrame.package_sets()

	# Initialize genetic algorithm population
	population = Population(size = population_size, cost_fn = ecnet_limit_inputs, select_fn = minimize_best_n)

	# Create genetic algorithm parameters for each input parameter *n*
	for i in range(limit_num):
		population.add_parameter(i, 0, DataFrame.num_inputs - 1)

	# Generate the genetic algorithm's initial population
	population.generate_population()

	# Print average population fitness (if printing feedback)
	if print_feedback:
		print('Generation: 0 - Population fitness: ' + str(sum(p.fitness_score for p in population.members) / len(population)))

	# Run the genetic algorithm for *num_generations* generations
	for gen in range(num_generations):
		population.next_generation(num_survivors = num_survivors, mut_rate = 0)
		if print_feedback:
			print('Generation: ' + str(gen + 1) + ' - Population fitness: ' + str(sum(p.fitness_score for p in population.members) / len(population)))

	# Find the best performing member from the final generation
	min_idx = 0
	for new_idx, member in enumerate(population.members):
		if member.fitness_score < population.members[min_idx].fitness_score:
			min_idx = new_idx

	# Obtain parameter names from best performer
	input_list = []
	for val in population.members[min_idx].feed_dict.values():
		input_list.append(DataFrame.input_names[val])

	# Print best performer (if printing feedback)
	if print_feedback:
		print('Best member fitness score: ' + str(population.members[min_idx].fitness_score))
		print(input_list)

	# Return the limited list
	return input_list

'''
Saves the parameters *param_list* (obtained from limit) to new database specified
by *filename*. A *DataFrame* object is required for new database formatting and
populating.
'''
def output(DataFrame, param_list, filename):

	# Check filename format
	if '.csv' not in filename:
		filename += '.csv'

	# List of rows to be saved to CSV file
	rows = []

	# FIRST ROW: type headers
	type_row = []
	type_row.append('DATAID')
	type_row.append('ASSIGNMENT')
	for string in DataFrame.string_names:
		type_row.append('STRING')
	for group in DataFrame.group_names:
		type_row.append('GROUP')
	for target in DataFrame.target_names:
		type_row.append('TARGET')
	for input_param in param_list:
		type_row.append('INPUT')
	rows.append(type_row)

	# SECOND ROW: titles (including string, group, target, input names)
	title_row = []
	title_row.append('DATAID')
	title_row.append('ASSIGNMENT')
	for string in DataFrame.string_names:
		title_row.append(string)
	for group in DataFrame.group_names:
		title_row.append(group)
	for target in DataFrame.target_names:
		title_row.append(target)
	for input_param in param_list:
		title_row.append(input_param)
	rows.append(title_row)

	# Obtain new parameter name indices in un-limited database
	input_param_indices = []
	for param in param_list:
		input_param_indices.append(DataFrame.input_names.index(param))

	# Create rows for each data point found in the DataFrame
	for point in DataFrame.data_points:
		data_row = []
		data_row.append(point.id)
		data_row.append(point.assignment)
		for string in point.strings:
			data_row.append(string)
		for group in point.groups:
			data_row.append(group)
		for target in point.targets:
			data_row.append(target)
		for param in input_param_indices:
			data_row.append(point.inputs[param])
		rows.append(data_row)

	# Save all the rows to the new database file
	with open(filename, 'w') as file:
		wr = csv.writer(file, quoting = csv.QUOTE_ALL, lineterminator = '\n')
		for row in rows:
			wr.writerow(row)