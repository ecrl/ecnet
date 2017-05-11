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

	rmse = server.test_model_rmse()			# Returns the RMSE of the ANN tested on the ENTIRE input database
	print(rmse)

	results = server.use_mlp_model()		# Uses the model on the current test set
	print(results)
