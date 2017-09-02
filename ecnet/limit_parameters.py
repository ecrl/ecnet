#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet_limit_parameters.py
#  
#  Developed in 2017 by Travis Kessler <Travis_Kessler@student.uml.edu>
#  
#  This program contains the functions necessary for reducing the input dimensionality of a database to the most influential input parameters
#

import csv
import copy

def limit(server, param_num):
	# 
	data = copy.copy(server.data)
	
	# Grabs paramter names
	param_names = data.param_cols[:]
	param_list = []
	
	# Exclude output parameters from algorithm
	for output in range(0,data.controls_num_outputs):
		del param_names[output]
		
	# Initial definition of total lists used for limiting
	learn_params = []
	valid_params = []
	total_params = []
	
	# Until the param_list is populated with specified number of params:	
	while len(param_list) < param_num:
		
		# Used for determining which paramter at the current iteration performs the best
		param_RMSE_list = []
		
		# Grabs each parameter one at a time
		for parameter in range(0,len(data.x[0])):
			
			# Each parameter is a sublist of the total lists
			learn_params_add = [sublist[parameter] for sublist in data.learn_x]
			valid_params_add = [sublist[parameter] for sublist in data.valid_x]
			total_params_add = [sublist[parameter] for sublist in data.x]
			
			# Formatting
			for i in range(0,len(learn_params_add)):
				learn_params_add[i] = [learn_params_add[i]]
			for i in range(0,len(valid_params_add)):
				valid_params_add[i] = [valid_params_add[i]]
			for i in range(0,len(total_params_add)):
				total_params_add[i] = [total_params_add[i]]
				
			# If looking for the first parameter, each parameter is tested individually
			if len(learn_params) is 0:
				learn_input = learn_params_add[:]
				valid_input = valid_params_add[:]
				total_input = total_params_add[:]
			
			# Else, new parameter in question is appended to the current parameter list
			else:
				learn_input = []
				valid_input = []
				total_input = []
				
				# Adds the current paramter lists to the inputs
				for i in range(0,len(learn_params)):
					learn_input.append(learn_params[i][:])
				for i in range(0,len(valid_params)):
					valid_input.append(valid_params[i][:])
				for i in range(0,len(total_params)):
					total_input.append(total_params[i][:])
				
				# Adds the new paramter in question	
				for i in range(0,len(learn_params_add)):
					learn_input[i].append(learn_params_add[i][0])
				for i in range(0,len(valid_params_add)):
					valid_input[i].append(valid_params_add[i][0])
				for i in range(0,len(total_params_add)):
					total_input[i].append(total_params_add[i][0])
			
			# Re-imports data for training
			#server.import_data()
			
			# Assigns the configured data to the server data object
			server.data.x = total_input[:]
			server.data.y = data.y[:]
			server.data.learn_x = learn_input[:]
			server.data.learn_y = data.learn_y[:]
			server.data.valid_x = valid_input[:]
			server.data.valid_y = data.valid_y[:]
			
			# Trains the model
			server.create_mlp_model()
			server.fit_mlp_model_validation()
			
			# Determines the RMSE of the model with the current inputs, adds it to total list
			local_rmse = server.calc_error('rmse')
			param_RMSE_list.append(local_rmse)
			
		# Determines lowest RMSE of the current iteration, which corresponds to the best performing parameter
		val, idx = min((val, idx) for (idx, val) in enumerate(param_RMSE_list))
		
		# Packages the best performing parameter
		add_to_learn = [sublist[idx] for sublist in data.learn_x]
		add_to_valid = [sublist[idx] for sublist in data.valid_x]
		add_to_total = [sublist[idx] for sublist in data.x]
		
		# Adds the best performing parameter to the total lists ***Conditional used for formatting discrepancies
		if len(param_list) is 0:
			for i in range(0,len(add_to_learn)):
				learn_params.append([add_to_learn[i]])
			for i in range(0,len(add_to_valid)):
				valid_params.append([add_to_valid[i]])
			for i in range(0,len(add_to_total)):
				total_params.append([add_to_total[i]])
		else:
			for i in range(0,len(add_to_learn)):
				learn_params[i].append(add_to_learn[i])
			for i in range(0,len(add_to_valid)):
				valid_params[i].append(add_to_valid[i])
			for i in range(0,len(add_to_total)):
				total_params[i].append(add_to_total[i])
				
		# Adds the best performing parameter to the parameter list
		param_list.append(param_names[idx])
				
		# Prints the parameter list after each iteration, as well as the RMSE
		if server.project_print_feedback == True:
			print(param_list)
			print(val)
		
	# Returns the parameter list
	return param_list
	
def output(data, param_list, filename):
	# Checks for .csv file format
	if ".csv" not in filename:
		filename = filename + ".csv"
	# Creates list of spreadsheet rows
	rows = []
	# Row 1: Main controls
	control_row_1 = ["NUM OF MASTER"]
	for i in range(0,len(data.controls_param_cols)):
		control_row_1.append(data.controls_param_cols[i])
	rows.append(control_row_1)
	# Row 2: Main control values
	control_row_2 = [data.controls_m_param_count]
	for i in range(0,len(data.control_params)):
		control_row_2.append(data.control_params[i])
	rows.append(control_row_2)
	# Rows 3 and 4: Column groups and sub-groups
	row_3 = ["DATAID", "T/V/L/U"]
	row_4 = ["DATAid", "T/V/L"]
	if data.controls_num_str != 0:
		row_3.append("STRINGS")
		for i in range(0,data.controls_num_str - 1):
			row_3.append(" ")
		for i in range(0,len(data.string_cols)):
			row_4.append(data.string_cols[i])
	if data.controls_num_grp != 0:
		row_3.append("GROUPS")
		for i in range(0,data.controls_num_grp - 1):
			row_3.append(" ")
		for i in range(0,len(data.group_cols)):
			row_4.append(data.group_cols[i])
	row_3.append("PARAMETERS")
	rows.append(row_3)
	for i in range(0,data.controls_num_outputs):
		row_4.append(data.param_cols[i])
	param_idx = []
	for i in range(0,len(param_list)):
		row_4.append(param_list[i])
		for j in range(0,len(data.param_cols)):
			if param_list[i] == data.param_cols[j]:
				param_idx.append(j)
				break
	rows.append(row_4)
	# Data value rows
	for i in range(0,len(data.dataid)):
		local_row = [data.dataid[i], data.tvl_strings[i]]
		for j in range(0,len(data.strings[i])):
			local_row.append(data.strings[i][j])
		for j in range(0,len(data.groups[i])):
			local_row.append(data.groups[i][j])
		for j in range(0,data.controls_num_outputs):
			local_row.append(data.params[i][j])
		for j in range(0,len(param_idx)):
			local_row.append(data.params[i][param_idx[j]])
		rows.append(local_row)
	# Output to file
	with open(filename, 'w') as output_file:
		wr = csv.writer(output_file, quoting = csv.QUOTE_ALL, lineterminator = '\n')
		for row in range(0,len(rows)):
			wr.writerow(rows[row])
