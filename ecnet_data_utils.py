#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet_data_utils.py
#  
#  Developed in 2017 by Travis Kessler <travis.j.kessler@gmail.com>
#  
#  This program contains the data object class, and functions for manipulating/importing/outputting data
#

import csv
import random
import pickle
import numpy as np
from math import sqrt
import sys

# Denormalizes resultant data using parameter file
def denormalize_result(results, param_filepath):
	normalParams = pickle.load(open(param_filepath + ".ecnet","rb"))
	for i in range(0,len(results[0])):
		for j in range(0,len(results)):
			results[j][i] = (results[j][i]*normalParams[i][1])-normalParams[i][0]
	return(results)

# Calculates the RMSE of the model result and the target data	
def calc_rmse(results, target):
	try:
		return(np.sqrt(((results-target)**2).mean()))
	except:
		try:
			return(np.sqrt(((np.asarray(results)-np.asarray(target))**2).mean()))
		except:
			print("Error in calculating RMSE. Check input data format.")
			sys.exit()

# Saves test results, data strings, groups to desired output .csv file			
def output_results(results, data, filename):
	# Makes sure filetype is csv
	if ".csv" not in filename:
		filename = filename + ".csv"
	# List of all rows
	rows = []
	# Title row, containing column names
	title_row = []
	title_row.append("DataID")
	for string in range(0,len(data.string_cols)):
		title_row.append(data.string_cols[string])
	for group in range(0,len(data.group_cols)):
		title_row.append(data.group_cols[group])
	title_row.append("DB Value")
	title_row.append("Predicted Value")
	rows.append(title_row)
	# Adds data ID's, strings, groups, DB values and predictions for each test result to the rows list
	for result in range(0,len(results[0])):
		local_row = []
		local_row.append(data.test_dataid[result])
		for string in range(0,len(data.test_strings[result])):
			local_row.append(data.test_strings[result][string])
		for group in range(0,len(data.test_groups[result])):
			local_row.append(data.test_groups[result][group])
		local_row.append(data.test_y[result][0])
		local_row.append(results[0][result][0])
		rows.append(local_row)
	# Outputs each row to the output file
	with open(filename, 'w') as output_file:
		wr = csv.writer(output_file, quoting = csv.QUOTE_ALL, lineterminator = '\n')
		for row in range(0,len(rows)):
			wr.writerow(rows[row])
	
### Initial definition of data object
class initialize_data:  
	def __init__(self, data_filename):
		self.file = data_filename

	# Opening excel (csv) file, and parsing initial data
	def build(self):
		if(".xlsx" in self.file):
			print(".xlsx file format detected. Please reformat as '.csv'.")
			sys.exit()
		elif(".csv" in self.file):
			with open(self.file, newline='') as csvfile:
				fileRaw = csv.reader(csvfile)
				fileRaw = list(fileRaw)
					# generates a raw list/2D Array for rows + cols of csv file;
					# i.e. cell A1 = [0][0], A2 = [1][0], B2 = [1][1], etc.
		else:
			print("Error: Unsupported file format")
			sys.exit()
			
		# parse master parameters from .csv file
		controls_m_param_count = int(fileRaw[1][0]) # num of master parameters, defined by A2
		controls_param_cols = fileRaw[0][1:1+controls_m_param_count] # ROW 1
		control_params = fileRaw[1][1:1+controls_m_param_count] # ROW 2
		controls_num_str = int(control_params[0])
		controls_num_grp = int(control_params[1])
		self.controls_num_outputs = int(control_params[4])
		
		# parse column names from .csv file
		self.string_cols = fileRaw[3][2:2+controls_num_str]
		self.group_cols = fileRaw[3][2+controls_num_str:2+controls_num_str+controls_num_grp]
		self.param_cols = fileRaw[3][2+controls_num_str+controls_num_grp:-1]
		(self.param_cols).append(fileRaw[3][-1])
		
		# parse data from .csv file
		self.dataid = [sublist[0] for sublist in fileRaw]
		del self.dataid[0:4] #removal of title rows
		self.strings = [sublist[2:2+controls_num_str] for sublist in fileRaw]
		del self.strings[0:4] #removal of title rows
		self.groups = [sublist[2+controls_num_str:2+controls_num_str+controls_num_grp] for sublist in fileRaw]
		del self.groups[0:4] #removal of title rows
		self.params = [sublist[2+controls_num_str+controls_num_grp:-1] for sublist in fileRaw]
		del self.params[0:4] #removal of title rows
		params_last = [sublist[-1] for sublist in fileRaw]
		del(params_last[0:4])
		for i in range(0,len(self.params)):
			self.params[i].append(params_last[i])
			
		# parse T/V/L data
		self.tvl_strings = [sublist[1] for sublist in fileRaw]
		del self.tvl_strings[0:4] #removal of title rows
		
		# Drop any data from data set defined in 'Data to AUTOMATICALLY DROP' or Unreliable in csv file
		dropListIndex = control_params[controls_param_cols.index("Data to AUTOMATICALLY DROP")]
		try:
			drop_remaining = dropListIndex.split( )
			while len(drop_remaining) != 0:
				dropRowNum = self.dataid.index(drop_remaining[0])
				del self.dataid[dropRowNum]
				del self.strings[dropRowNum]
				del self.groups[dropRowNum]
				del self.params[dropRowNum]
				del self.tvl_strings[dropRowNum]
				del drop_remaining[0]
		except:
			pass
		self.unreliable = []
		for i in range(0,len(self.dataid)):
			if (self.tvl_strings[i]).startswith("U"): # deleting predetermined unreliable data
				(self.unreliable).append(i)
		for i in range(0,len(self.unreliable)):
			del self.dataid[self.unreliable[-1]]
			del self.strings[self.unreliable[-1]]
			del self.groups[self.unreliable[-1]]
			del self.params[self.unreliable[-1]]
			del self.tvl_strings[self.unreliable[-1]]
			del self.unreliable[-1]
		# End of building

	# Normalizing the parameter data to be within range [0,1] for each parameter. For use with sigmoidal activation functions only.
	def normalize(self, param_filepath = 'normalParams'):
		minMaxList = []
		for i in range(0,len(self.params[0])):
			beforeNormal = [sublist[i] for sublist in self.params]
			beforeNormal = np.matrix(beforeNormal).astype(np.float).reshape(-1,1)
			minVal = beforeNormal.min()
			minAdjust = 0 - minVal # IMPORTANT VARIABLE for de-normalizing predicted data into final predicted format
			for a in range(0,len(beforeNormal)):
				beforeNormal[a] = beforeNormal[a] + minAdjust
			maxVal = beforeNormal.max() # IMPORTANT VARIABLE for de-normalizing predicted data into final predicted format
			minMaxList.append([minAdjust,maxVal])
			for b in range(0,len(beforeNormal)):
				beforeNormal[b] = beforeNormal[b]/maxVal
			if i is 0:
				normalized_list = beforeNormal
				maxVal = maxVal
			else:
				normalized_list = np.column_stack([normalized_list,beforeNormal])
		normalized_list = normalized_list.tolist()
		self.params = normalized_list
		pickle.dump(minMaxList,open(param_filepath + ".ecnet","wb")) # Saves the parameter list for opening in data normalization and predicting

	# Applying normalizing parameters using parameter file to new (unseen) data. Based on previously build network.
	def applyNormal(self, param_filepath = 'normalParams'):
		self.normalParams = pickle.load(open(param_filepath + ".ecnet","rb"))
		inputParams = []
		outputParams = []
		for i in range(0,len(self.params)):
			paramBeforeNorm = []
			for j in range(0,len(self.params[i])):
				paramBeforeNorm.append(float(self.params[i][j]))
			inputParamsAdd = []
			for j in range(0,len(paramBeforeNorm)):
				if j == 0:
					outputParams.append((paramBeforeNorm[j] + self.normalParams[j][0])/self.normalParams[j][1])
				else:
					inputParamsAdd.append((paramBeforeNorm[j] + self.normalParams[j][0])/self.normalParams[j][1])
			inputParams.append(inputParamsAdd)

	# Defining data for test, validation or learning based on either imported file or randomization
	def buildTVL(self, sort_type = 'random', data_split = [0.65, 0.25, 0.1]):
		self.testIndex = []
		self.validIndex = []
		self.learnIndex = []        
		if 'explicit' in sort_type:
			for i in range(0,len(self.dataid)):
				if (self.tvl_strings[i]).startswith("T"):
					(self.testIndex).append(i)
				if (self.tvl_strings[i]).startswith("V"):
					(self.validIndex).append(i)
				if (self.tvl_strings[i]).startswith("L"):
					(self.learnIndex).append(i)
		elif 'random' in sort_type:
			randIndex = random.sample(range(len(self.dataid)),len(self.dataid))
			self.randomizeData(randIndex, data_split)
		else:
			print("Error: unknown sort_type method, no splitting done.")
			sys.exit()

	# Randomizes T/V/L lists based on splitting percentage variables
	def randomizeData(self, randIndex, data_split):
		if data_split[2] != 0:
			if data_split[2] != 1:
				for i in range(0,round(data_split[2]*len(randIndex))):
					(self.testIndex).append(randIndex[i])
					del randIndex[i]
			else:
				for i in range(0,len(randIndex)):
					(self.testIndex).append(randIndex[i])
				randIndex = []
			(self.testIndex).sort()
		if data_split[1] != 0:
			if data_split[1] != 1:
				for i in range(0,round(data_split[1]*len(randIndex))):
					(self.validIndex).append(randIndex[i])
					del randIndex[i]
			else:
				for i in range(0,len(randIndex)):
					(self.validIndex).append(randIndex[i])
				randIndex = []
			(self.validIndex).sort()
		self.learnIndex = randIndex
		(self.learnIndex).sort()

	# Application of index values to data
	def applyTVL(self):
		self.test_dataid = []
		self.test_params = []
		self.test_strings = []
		self.test_groups = []
		self.valid_dataid = []
		self.valid_params = []
		self.valid_strings = []
		self.valid_groups = []
		self.learn_dataid = []
		self.learn_params = []
		self.learn_strings = []
		self.learn_groups = []
		for i in range(0,len(self.testIndex)):
			(self.test_dataid).append(self.dataid[self.testIndex[i]])
			(self.test_params).append(self.params[self.testIndex[i]])
			(self.test_strings).append(self.strings[self.testIndex[i]])
			(self.test_groups).append(self.groups[self.testIndex[i]])
		for i in range(0,len(self.validIndex)):
			(self.valid_dataid).append(self.dataid[self.validIndex[i]])
			(self.valid_params).append(self.params[self.validIndex[i]])
			(self.valid_strings).append(self.strings[self.validIndex[i]])
			(self.valid_groups).append(self.groups[self.validIndex[i]])
		for i in range(0,len(self.learnIndex)):
			(self.learn_dataid).append(self.dataid[self.learnIndex[i]])
			(self.learn_params).append(self.params[self.learnIndex[i]])
			(self.learn_strings).append(self.strings[self.learnIndex[i]])
			(self.learn_groups).append(self.groups[self.learnIndex[i]])

	# Builds x & y matrices (output for regression)
	# Applies to whole data set, plus TVL lists
	def package(self):
		if not('Result (Exp)' or 'Result' in self.param_cols[0]):
			print("Error: Improper indexing; check database format.")
			sys.exit()

		# Whole data set
		self.y = [sublist[0:self.controls_num_outputs] for sublist in self.params]
		self.x = [sublist[self.controls_num_outputs:-1] for sublist in self.params]
		x_last = [sublist[-1] for sublist in self.params]
		for i in range(0,len(self.x)):
			self.x[i].append(x_last[i])
		self.x = (np.asarray(self.x)).astype(np.float32)
		if self.controls_num_outputs > 1:
			for i in range(0,len(self.y)):
				for j in range(0,len(self.y[i])):
					self.y[i][j] = float(self.y[i][j])
		else:
			self.y = (np.asarray(self.y)).astype(np.float32)
		
		# Test data set
		self.test_y = [sublist[0:self.controls_num_outputs] for sublist in self.test_params]
		self.test_x = [sublist[self.controls_num_outputs:-1] for sublist in self.test_params]
		test_x_last = [sublist[-1] for sublist in self.test_params]
		for i in range(0,len(self.test_x)):
			self.test_x[i].append(test_x_last[i])
		self.test_x = (np.asarray(self.test_x)).astype(np.float32)
		if self.controls_num_outputs > 1:
			for i in range(0,len(self.test_y)):
				for j in range(0,len(self.test_y[i])):
					self.test_y[i][j] = float(self.test_y[i][j])
		else:
			self.test_y = (np.asarray(self.test_y)).astype(np.float32)
		
		# Validation data set
		self.valid_y = [sublist[0:self.controls_num_outputs] for sublist in self.valid_params]
		self.valid_x = [sublist[self.controls_num_outputs:-1] for sublist in self.valid_params]
		valid_x_last = [sublist[-1] for sublist in self.valid_params]
		for i in range(0,len(self.valid_x)):
			self.valid_x[i].append(valid_x_last[i])
		self.valid_x = (np.asarray(self.valid_x)).astype(np.float32)
		if self.controls_num_outputs > 1:
			for i in range(0,len(self.valid_y)):
				for j in range(0,len(self.valid_y[i])):
					self.valid_y[i][j] = float(self.valid_y[i][j])
		else:
			self.valid_y = (np.asarray(self.valid_y)).astype(np.float32)
		
		# Learning data set
		self.learn_y = [sublist[0:self.controls_num_outputs] for sublist in self.learn_params]
		self.learn_x = [sublist[self.controls_num_outputs:-1] for sublist in self.learn_params]
		learn_x_last = [sublist[-1] for sublist in self.learn_params]
		for i in range(0,len(self.learn_x)):
			self.learn_x[i].append(learn_x_last[i])
		self.learn_x = (np.asarray(self.learn_x)).astype(np.float32)
		if self.controls_num_outputs > 1:
			for i in range(0,len(self.learn_y)):
				for j in range(0,len(self.learn_y[i])):
					self.learn_y[i][j] = float(self.learn_y[i][j])
		else:
			self.learn_y = (np.asarray(self.learn_y)).astype(np.float32)

        
