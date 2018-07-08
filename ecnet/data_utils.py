#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet/data_utils.py
#  v.1.4.3
#  Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
#  This program contains the "DataFrame" class, and functions for processing/importing/outputting
#	data. High-level usage is handled by the "Server" class in server.py. For low-level
#	usage explanations, refer to https://github.com/tjkessler/ecnet
#

# 3rd party packages (open src.)
import csv
import numpy as np
import random as rm

'''
Private DataPoint class: Contains all information for each data entry found in CSV database
'''
class DataPoint:

	def __init__(self):

		self.id = None
		self.assignment = None
		self.strings = []
		self.groups = []
		self.targets = []
		self.inputs = []

'''
DataFrame class: Handles importing data from formatted CSV database, determining learning,
validation and testing sets, and packages sets as Numpy arrays for hand-off to models
'''
class DataFrame:

	'''
	Initializes object, creates *DataPoint*s for each data entry
	'''
	def __init__(self, filename):

		# Make sure filename is CSV
		if not '.csv' in filename:
			filename = filename + '.csv'
		# Open the database file
		try:
			with open(filename, newline = '') as file:
				data_raw = csv.reader(file)
				data_raw = list(data_raw)
		# Database not found!
		except FileNotFoundError:
			raise Exception('ERROR: Supplied file not found in working directory')

		# Append each database data point to DataFrame's data_point list
		self.data_points = []
		for point in range(2, len([sublist[0] for sublist in data_raw])):
			# Define data point
			new_point = DataPoint()
			# Set data point's id
			new_point.id = [sublist[0] for sublist in data_raw][point]
			# Set data point's assignment
			new_point.assignment = [sublist[1] for sublist in data_raw][point]
			for header in range(len(data_raw[0])):
				# Append data point strings
				if 'STRING' in data_raw[0][header]:
					new_point.strings.append(data_raw[point][header])
				# Append data point groups
				elif 'GROUP' in data_raw[0][header]:
					new_point.groups.append(data_raw[point][header])
				# Append data point target values
				elif 'TARGET' in data_raw[0][header]:
					new_point.targets.append(data_raw[point][header])
				# Append data point input values
				elif 'INPUT' in data_raw[0][header]:
					new_point.inputs.append(data_raw[point][header])

			# Append to data_point list
			self.data_points.append(new_point)

		# Obtain string, group, target, input header names
		self.string_names = []
		self.group_names = []
		self.target_names = []
		self.input_names = []
		for header in range(len(data_raw[0])):
			if 'STRING' in data_raw[0][header]:
				self.string_names.append(data_raw[1][header])
			if 'GROUP' in data_raw[0][header]:
				self.group_names.append(data_raw[1][header])
			if 'TARGET' in data_raw[0][header]:
				self.target_names.append(data_raw[1][header])
			if 'INPUT' in data_raw[0][header]:
				self.input_names.append(data_raw[1][header])

		# Helper variables for determining number of strings, groups, targets, inputs
		self.num_strings = len(self.string_names)
		self.num_groups = len(self.group_names)
		self.num_targets = len(self.target_names)
		self.num_inputs = len(self.input_names)

	'''
	DataFrame class length = number of data_points
	'''
	def __len__(self):

		return len(self.data_points)

	'''
	Creates learning, validation and test sets
	random: *True* for random assignments, *False* for explicit (point defined) assignments
	split: if random == *True*, split[0] = learning, split[1] = validation, split[2] = test 
	(proportions)
	'''
	def create_sets(self, random = True, split = [0.7, 0.2, 0.1]):

		# Define sets
		self.learn_set = []
		self.valid_set = []
		self.test_set = []

		# If using random assignments
		if random:
			# Create random indices for sets
			rand_index = rm.sample(range(len(self)), len(self))
			learn_index = rand_index[0 : int(len(rand_index) * split[0])]
			valid_index = rand_index[int(len(rand_index) * split[0]) : int(len(rand_index) * (split[0] + split[1]))]
			test_index = rand_index[int(len(rand_index) * (split[0] + split[1])) : -1]
			test_index.append(rand_index[-1])

			# Append data points to sets, set assignment to RAND (random assignment)
			for idx in learn_index:
				self.data_points[idx].assignment = 'L'
				self.learn_set.append(self.data_points[idx])
			for idx in valid_index:
				self.data_points[idx].assignment = 'V'
				self.valid_set.append(self.data_points[idx])
			for idx in test_index:
				self.data_points[idx].assignment = 'T'
				self.test_set.append(self.data_points[idx])

		# Using explicit (point defined) assignments
		else:
			# Append data points to sets based on explicit assignments
			for point in self.data_points:
				if point.assignment == 'L':
					self.learn_set.append(point)
				elif point.assignment == 'V':
					self.valid_set.append(point)
				elif point.assignment == 'T':
					self.test_set.append(point)

	'''
	Creates learning, validation and test sets containing specified proportions
	(*split*) of each *sort_string* element (*sort_string* can be any STRING value
	found in your database file)
	'''
	def create_sorted_sets(self, sort_string, split = [0.7, 0.2, 0.1]):
	
		# Obtain index of *sort_string* from DataFrame's string names
		string_idx = self.string_names.index(sort_string)
		# Sort DataPoints by specified string
		self.data_points.sort(key = lambda x: x.strings[string_idx])

		# List containing all possible values from *sort_string* string
		string_vals = []
		# Groups for each distinct string value, containing DataPoints
		string_groups = []

		# Find all string values in *sort_string*, add/create string_val and string_group entries
		for point in self.data_points:
			if point.strings[string_idx] not in string_vals:
				string_vals.append(point.strings[string_idx])
				string_groups.append([point])
			else:
				string_groups[-1].append(point)

		# Reset lists for new set splits
		self.learn_set = []
		self.valid_set = []
		self.test_set = []

		# For each distinct string value from *sort_string*:
		for group in string_groups:
			# Assign learning data
			learn_stop = int(split[0] * len(group))
			for point in group[0 : learn_stop]:
				point.assignment = 'L'
				self.learn_set.append(point)
			# Assign validation data
			valid_stop = learn_stop + int(split[1] * len(group))
			for point in group[learn_stop : valid_stop]:
				point.assignment = 'V'
				self.valid_set.append(point)
			# Assign testing data
			for point in group[valid_stop :]:
				point.assignment = 'T'
				self.test_set.append(point)

	'''
	Shuffles (new random assignments) the specified sets in *args*; (learning, validation, testing)
	or (learning, validation)
	'''
	def shuffle(self, *args, split = [0.7, 0.2, 0.1]):

		# Shuffle all sets (can just call create_sets again)
		if 'l' and 'v' and 't' in args:
			self.create_sets(split = split)

		# Shuffle training data (learning and validation sets)
		elif 'l' and 'v' in args:
			# Compile all training data into one list
			lv_set = []
			for point in self.learn_set:
				lv_set.append(point)
			for point in self.valid_set:
				lv_set.append(point)
			# Generate random indices for new learning and validation sets
			rand_index = rm.sample(range(len(self.learn_set) + len(self.valid_set)), (len(self.learn_set) + len(self.valid_set)))
			learn_index = rand_index[0 : int(len(rand_index) * (split[0] / (1 - split[2]))) + 1]
			valid_index = rand_index[int(len(rand_index) * (split[0] / (1 - split[2]))) + 1 : -1]
			valid_index.append(rand_index[-1])

			# Clear current learning and validation sets
			self.learn_set = []
			self.valid_set = []

			# Apply new indices to compiled training data, creating learning and validation sets
			for idx in learn_index:
				self.learn_set.append(lv_set[idx])
			for idx in valid_index:
				self.valid_set.append(lv_set[idx])
		else:
			raise Exception('ERROR: Shuffle arguments must be *l, v, t* or *l, v*')

	'''
	Private object containing lists (converted to numpy arrays by package_sets) with target 
	(y) and input (x) values (filled by package_sets)
	'''
	class __PackagedData:

		def __init__(self):

			self.learn_x = []
			self.learn_y = []
			self.valid_x = []
			self.valid_y = []
			self.test_x = []
			self.test_y = []

	'''
	Creates and returns *PackagedData* object containing numpy arrays with target (y) and 
	input (x) values for learning, validation and testing sets
	'''
	def package_sets(self):

		# Create PackagedData object to return
		pd = self.__PackagedData()
		# Append learning inputs, learning targets to PackagedData object
		for point in self.learn_set:
			pd.learn_x.append(np.asarray(point.inputs).astype('float32'))
			pd.learn_y.append(np.asarray(point.targets).astype('float32'))
		# Append validation inputs, validation targets to PackagedData object
		for point in self.valid_set:
			pd.valid_x.append(np.asarray(point.inputs).astype('float32'))
			pd.valid_y.append(np.asarray(point.targets).astype('float32'))
		# Append testing inputs, testing targets to PackagedData object
		for point in self.test_set:
			pd.test_x.append(np.asarray(point.inputs).astype('float32'))
			pd.test_y.append(np.asarray(point.targets).astype('float32'))
		# Lists -> Numpy arrays
		pd.learn_x = np.asarray(pd.learn_x)
		pd.learn_y = np.asarray(pd.learn_y)
		pd.valid_x = np.asarray(pd.valid_x)
		pd.valid_y = np.asarray(pd.valid_y)
		pd.test_x = np.asarray(pd.test_x)
		pd.test_y = np.asarray(pd.test_y)
		# Return packaged data
		return pd

'''
Outputs *results* to *filename*; *DataFrame*, a 'DataFrame' object, is required for
header formatting (strings, groups) and outputting individual point data
(id, assignment, strings, groups)
'''
def output_results(results, DataFrame, filename):

	# Ensure save filepath is CSV
	if '.csv' not in filename:
		filename += '.csv'

	# List of rows to be saved to CSV file
	rows = []

	# FIRST ROW: type headers
	type_row = []
	type_row.append('DATAID')
	type_row.append('ASSIGNMENT')
	for string in range(DataFrame.num_strings):
		type_row.append('STRING')
	for group in range(DataFrame.num_groups):
		type_row.append('GROUP')
	for target in range(DataFrame.num_targets):
		type_row.append('TARGET')
	for result in results:
		type_row.append('RESULT')
	rows.append(type_row)

	# SECOND ROW: titles (including string, group, target names)
	title_row = []
	title_row.append('DATAID')
	title_row.append('ASSIGNMENT')
	for string in DataFrame.string_names:
		title_row.append(string)
	for group in DataFrame.group_names:
		title_row.append(group)
	for target in DataFrame.target_names:
		title_row.append(target)
	for result in range(len(results)):
		title_row.append(result)
	rows.append(title_row)

	# Check which data set the results are from
	if len(results[0]) == len(DataFrame.learn_set):
		dset = 'learn'
	elif len(results[0]) == len(DataFrame.valid_set):
		dset = 'valid'
	elif len(results[0]) == len(DataFrame.test_set):
		dset = 'test'
	elif len(results[0]) == (len(DataFrame.learn_set) + len(DataFrame.valid_set)):
		dset = 'train'
	else:
		dset = None

	# If results are for training data, compile learning and validation data
	if dset == 'train':
		output_points = []
		for point in DataFrame.learn_set:
			output_points.append(point)
		for point in DataFrame.valid_set:
			output_points.append(point)
	# If results are for learning data, compile learning data
	elif dset == 'learn':
		output_points = DataFrame.learn_set
	# If results are for validation data, compile validation data
	elif dset == 'valid':
		output_points = DataFrame.valid_set
	# If results are for testing data, compile testing data
	elif dset == 'test':
		output_points = DataFrame.test_set
	# Else, assume results are for all data, compile learning, validation and testing data
	else:
		output_points = []
		for point in DataFrame.learn_set:
			output_points.append(point)
		for point in DataFrame.valid_set:
			output_points.append(point)
		for point in DataFrame.test_set:
			output_points.append(point)

	# Create rows for each data point in the compiled results
	for idx, point in enumerate(output_points):
		data_row = []
		data_row.append(point.id)
		data_row.append(point.assignment)
		for string in point.strings:
			data_row.append(string)
		for group in point.groups:
			data_row.append(group)
		for target in point.targets:
			data_row.append(target)
		for result in results:
			if DataFrame.num_targets == 1:
				data_row.append(result[idx][0])
			else:
				data_row.append(result[idx])
		rows.append(data_row)

	# Save rows to CSV file specified in *filename*
	with open(filename, 'w') as file:
		wr = csv.writer(file, quoting = csv.QUOTE_ALL, lineterminator = '\n')
		for row in rows:
			wr.writerow(row)