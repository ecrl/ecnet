#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet_data_utils.py
#  
#  Developed in 2017 by Travis Kessler <travis.j.kessler@gmail.com>
#  
#  This program contains functions necessary for manipulating and shaping a data object for a server to serve a neural network model
#

import csv
import random
import pickle
import numpy as np
from math import sqrt
import sys

# Denormalizes resultant data using parameter file
def denormalize_result(results, server):
	param_filepath = server.normal_params_filename
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
	
### SERVER CLASS OBJECT: Initial definition of data object
class initialize_data:  
	def __init__(self, server):
		self.file = server.data_filename

	# Opening excel (csv) file, and parsing initial data
	def build(self, server):
		nz = server.nz
		if(nz > 1):
			print("Running program using '" + self.file + "':")
			print("")
			print("Building list structure:")
		if(".xlsx" in self.file):
			print(".xlsx file format detected. Please reformat as '.csv'.")
			sys.exit()
		elif(".csv" in self.file):
			if(nz > 1):
				print("     Opening file...")
			#try:
			with open(self.file, newline='') as csvfile:
				fileRaw = csv.reader(csvfile)
				fileRaw = list(fileRaw)
					# generates a raw list/2D Array for rows + cols of csv file;
					# i.e. cell A1 = [0][0], A2 = [1][0], B2 = [1][1], etc.
			#except:
			#	print("Error: File not found")
			#	return
		else:
			print("Error: Unsupported file format")
			sys.exit()
		if(nz > 1):
			print("     Parsing information...")
			
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
				if (nz > 1):
					print("          Dropped data point "+drop_remaining[0])
				del drop_remaining[0]
			if(nz > 1):
				print("     Dropping of excluded data complete.")
		except:
			if(nz > 1):
				print("        No data dropped.")
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
		if nz > 1:
			print("Build complete.")
			print("")

	# Normalizing the parameter data to be within range [0,1] for each parameter. For use with sigmoidal activation functions only.
	def normalize(self, server):
		nz = server.nz
		param_filepath = server.normal_params_filename
		if nz > 1:
			print("Normalizing data:")
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
		if nz > 1:
			print("Data normalized; parameters output to '%s'")%(param_filepath + ".ecnet")

	# Applying normalizing parameters using 'normalParams.p' file to new (unseen) data. Based on previously build network.
	def applyNormal(self, server):
		nz = server.nz
		param_filepath = server.normal_params_filename
		if nz > 1:
			print("")
			print("Normalizing new data using '%s':")%(param_filepath + ".ecnet")
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
		if nz > 1:
				print("Normalizing complete.")

	# Defining data for test, validation or learning based on either imported file or randomization
	def buildTVL(self, server):
		sort_type = server.sort_type
		data_split = server.data_split
		nz = server.nz
		if(nz > 1):
			print("Building T/V/L/U lists:")
		self.testIndex = []
		self.validIndex = []
		self.learnIndex = []        
		if 'explicit' in sort_type:
			if(nz > 1):
				print("     Applying explicit T/V/L/U index values...")
			for i in range(0,len(self.dataid)):
				if (self.tvl_strings[i]).startswith("T"):
					(self.testIndex).append(i)
				if (self.tvl_strings[i]).startswith("V"):
					(self.validIndex).append(i)
				if (self.tvl_strings[i]).startswith("L"):
					(self.learnIndex).append(i)
			if(nz > 1):
				print("     Application complete.")
		elif 'random' in sort_type:
			if(nz > 1):
				print("     Applying random T/V/L index values...")
			randIndex = random.sample(range(len(self.dataid)),len(self.dataid))
			self.randomizeData(randIndex, data_split)
			if(nz > 1):
				print("     Application complete.")
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
	def applyTVL(self, server):
		nz = server.nz
		if(nz > 1):
			print("     Applying index values to data...")
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
		if(nz > 1):
			print("     Application Complete.")
			print("Build Complete.")
			print("")

	# Builds x & y matrices (output for regression)
	# Applies to whole data set, plus TVL lists
	def package(self, server):
		nz = server.nz
		if(nz > 1):
			print("Packaging data for output:")
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
		
		#self.use_conf_weights = False
		
		if(nz > 1):
			print("Packaging complete.")
			print("")
			print("Number of initial data values:")
			print(len(self.dataid))

        
