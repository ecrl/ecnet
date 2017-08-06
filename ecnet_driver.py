#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet_driver.py
#  
#  Developed in 2017 by Travis Kessler <Travis_Kessler@student.uml.edu>
#
#  This program shows examples of project/neural network creation, using ecnet_server.py and config.yml
#

import ecnet_server

# Example for creating a project, importing data, creating, fitting and evaluating the model, and outputting results:
def create_project_example():
	server = ecnet_server.Server()			# Creates the Server
	server.create_save_env()				# Creates the folder structure for output files; if not used, a single neural network is saved to the working directory
	server.import_data()					# Imports the data to the Server
	server.create_mlp_model()				# Creates the ANN skeleton based on config variables
	server.fit_mlp_model_validation()		# Fits the ANN to the data using validation; use "fit_mlp_model()" for non-validation use
	server.select_best()					# Selects the best trial from each node for each build; must have used "create_save_env()"

	rmse = server.test_model_rmse()			# Returns the RMSE of the model(s) tested on the ENTIRE input database
	print(rmse)
	
	r2 = server.test_model_r2()				# Returns the coefficient of determination (r-squared) of the model(s) tested on the ENTIRE input database
	print(r2)
	
	mae = server.test_model_mae()			# Returns the MAE of the model(s) tested on the ENTIRE input database
	print(mae)

	results = server.use_mlp_model_all()	# Uses the model on the current data set; use "use_mlp_model()" to grab results just for the test set
	
	server.output_results(results, "all", "on_mon_v1_a_2.csv")			# Outputs results to a specified file; "all" for "use_mlp_model_all()", "test" for "use_mlp_model()"

# Example for reducing the input dimensionality of a large database to the most influential input parameters, and outputting a new database:
def limit_parameters_example():
	server = ecnet_server.Server()			# Creates the Server
	server.import_data()					# Imports the data to the Server
	server.limit_parameters("new_database_filename.csv")	# Reduces input dimensionality of parameters to 'param_limit_num' in config, outputs a new database to specified file

if __name__ == "__main__":
	create_project_example()
