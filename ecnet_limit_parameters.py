import ecnet_server

def limit(data, num_params):
	
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
	while len(param_list) < num_params:
		
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
			
			# Creates an ecnet_server for training
			limit_server = ecnet_server.Server()
			limit_server.import_data()
			
			# Assigns the configured data to the server data object
			limit_server.data.x = total_input[:]
			limit_server.data.y = data.y[:]
			limit_server.data.learn_x = learn_input[:]
			limit_server.data.learn_y = data.learn_y[:]
			limit_server.data.valid_x = valid_input[:]
			limit_server.data.valid_y = data.valid_y[:]
			
			# Trains the model
			limit_server.create_mlp_model()
			limit_server.fit_mlp_model_validation()
			
			# Determines the RMSE of the model with the current inputs, adds it to total list
			local_rmse = limit_server.test_model_rmse()
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
		print(param_list)
		print(val)
		
	# Returns the parameter list
	return param_list
			

### Example function call ###
if __name__ == "__main__":
	server = ecnet_server.Server()
	server.import_data()
	limit(server.data, server.param_limit_num)
