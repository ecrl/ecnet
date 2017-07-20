#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet_driver.py
#  
#  Developed in 2017 by Travis Kessler <travis.j.kessler@gmail.com>
#
#  This program handles the flow of neural network creation, using ecnet_server.py
#

import ecnet_server

if __name__ == "__main__":

	server = ecnet_server.Server()			# Creates the Server
	server.import_data()					# Imports the data to the Server
	server.create_save_env()				# Creates the folder structure for output files; if not used, a single ANN is saved to the working directory
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
	
	server.output_results(results, "all", "my_results.csv")			# Outputs results to a specified file; "all" for "use_mlp_model_all()", "test" for "use_mlp_model()"
