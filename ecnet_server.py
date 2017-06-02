#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet_server.py
#  
#  Developed in 2017 by Travis Kessler <travis.j.kessler@gmail.com>
#
#  This program contains all the necessary config parameters and network serving functions
#

# 3rd party packages (open src.)
import yaml
import numpy as np
import sys
import os

# KNET program files
import ecnet_data_utils
import ecnet_model
from ecnet_model import multilayer_perceptron

### Config/server object; to be referenced by most other files ###
class Server:
    ### Initial declaration, handles config import
	def __init__(self):
		self.__dict__.update(import_config())
		self.folder_structs_built = False

	### Imports the data and stores it in the server	
	def import_data(self):
		try:
			self.data = ecnet_data_utils.initialize_data(self.data_filename)
		except:
			print("Error: Cannot find 'data_filename' in 'config.yml'.")
			raise
			sys.exit()
		try:
			self.data.build()
		except:
			print("Error: Unable to parse file. Check file format for unity.")
			raise
			sys.exit()
		try:
			self.data.buildTVL(self.sort_type, self.data_split)
			if self.normals_use == True:
				self.data.normalize(self.normal_params_filename)
			self.data.applyTVL()
			self.data.package()
		except:
			print("Error: Unable to build data TVL. Check syntax.")
			raise
			sys.exit()
	
	### NOT FUNCTIONAL - TODO: Add function for limiting parameters to data utils file		
	def limit_parameters(self):
		return
		#return ecnet_data_utils.limit_parameters(self.data, self.param_limit_num)

	### Creates the save environment
	def create_save_env(self):
		try:
			create_folder_structure(self)
		except:
			print("Error: Unknown error in creating project folder structure.")
			raise
			sys.exit()

	### Creates the model once the data is loaded		
	def create_mlp_model(self):
		try:
			self.model = create_model(self)
		except:
			print("Error: Unable to create model. Make sure data has been loaded and/or check 'config.yaml' for syntax errors with structure parameters.")
			raise
			sys.exit()
	
	### Fits the model using predetermined number of learning epochs	
	def fit_mlp_model(self):
        ### MULTINET ###
		if self.folder_structs_built == True:
			for build in range(0,self.num_builds):
				print("Build %d of %d"%(build+1,self.num_builds))
				for node in range(0,self.num_nodes):
					print("Node %d of %d"%(node+1,self.num_nodes))
					for trial in range(0,self.num_trials):
						print("Trial %d of %d"%(trial+1,self.num_trials))
						self.output_filepath = os.path.join(os.path.join(os.path.join(self.project_name, self.build_dirs[build]), self.node_dirs[build][node]),self.model_output_filename + "_%d"%(trial + 1))
						try:
							self.model.fit(self.data.learn_x, self.data.learn_y, self.learning_rate, self.train_epochs)
						except:
							print("Error: model must be created before it can be fit.")
							raise
							sys.exit()
						try:
							self.model.save_net(self.output_filepath)
						except:
							print("Error: Unable to save model. Check 'config.yaml' for filename mismatches.")
							raise
							sys.exit()
						self.import_data()
						self.create_mlp_model()
		### SINGLE NET ###
		else:
			self.output_filepath = self.model_output_filename
			try:
				self.model.fit(self.data.learn_x, self.data.learn_y, self.learning_rate, self.train_epochs)
			except:
				print("Error: model must be created before it can be fit.")
				raise
				sys.exit()
			try:
				self.model.save_net(self.output_filepath)
			except:
				print("Error: Unable to save model. Check 'config.yaml' for filename mismatches.")
				raise
				sys.exit()

	### Fits the model using validation RMSE cutoff method		
	def fit_mlp_model_validation(self):
		### MULTINET ###
		if self.folder_structs_built == True:
			for build in range(0,self.num_builds):
				print("Build %d of %d"%(build+1,self.num_builds))
				for node in range(0,self.num_nodes):
					print("Node %d of %d"%(node+1,self.num_nodes))
					for trial in range(0,self.num_trials):
						print("Trial %d of %d"%(trial+1,self.num_trials))
						self.output_filepath = os.path.join(os.path.join(os.path.join(self.project_name, self.build_dirs[build]), self.node_dirs[build][node]), self.model_output_filename + "_%d"%(trial + 1))
						try:
							self.model.fit_validation(self.data.learn_x, self.data.valid_x, self.data.learn_y, self.data.valid_y, self.learning_rate, self.valid_mdrmse_stop, self.valid_mdrmse_memory, self.valid_max_epochs)
						except:
							print("Error: model must be created before it can be fit.")
							raise
							sys.exit()
						try:
							self.model.save_net(self.output_filepath)
						except:
							print("Error: Unable to save model. Check 'config.yaml' for filename mismatches.")
							raise
							sys.exit()
						self.import_data()
						self.create_mlp_model()
		### SINGLE NET ###
		else:
			self.output_filepath = self.model_output_filename
			try:
				self.model.fit_validation(self.data.learn_x, self.data.valid_x, self.data.learn_y, self.data.valid_y, self.learning_rate, self.valid_mdrmse_stop, self.valid_mdrmse_memory, self.valid_max_epochs)
			except:
				print("Error: model must be created before it can be fit.")
				raise
				sys.exit()
			try:
				self.model.save_net(self.output_filepath)
			except:
				print("Error: Unable to save model. Check 'config.yaml' for filename mismatches.")
				raise
				sys.exit()

	### Selects the best performing networks from each node of each build. Folder structs must be created.			
	def select_best(self):
		### SINGLE NET ###
		if self.folder_structs_built == False:
			print("Error: Folder structs must be built in order to select best.")
			sys.exit()
		### MULTINET ###
		else:
			for i in range(0,self.num_builds):
				for j in range(0,self.num_nodes):
					rmse_list = []
					for k in range(0,self.num_trials):
						self.model_load_filename = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),self.model_output_filename + "_%d"%(k+1)))
						self.model = ecnet_model.multilayer_perceptron()
						self.model.load_net(self.model_load_filename)
						res = self.model.test_new(self.data.x)
						rmse = ecnet_data_utils.calc_rmse(res, self.data.y)
						rmse_list.append(rmse)
					current_min = 0
					for error in range(0,len(rmse_list)):
						if rmse_list[error] < rmse_list[current_min]:
							current_min = error
					self.model_load_filename = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),self.model_output_filename + "_%d"%(current_min+1)))
					self.output_filepath = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),"final_net_%d"%(j+1)))
					self.resave_net(self.output_filepath)
	
	def use_mlp_model(self):
		### SINGLE NET ###
		if self.folder_structs_built == False:
			self.model = ecnet_model.multilayer_perceptron()
			try:
				self.model.load_net(self.model_load_filename)
			except:
				print("Error: Unable to load model; one must be created first. If one exists, check 'condig.yaml' for filename mismatches.")
				raise
				sys.exit()
			try:
				if self.normals_use == True:
					res = ecnet_data_utils.denormalize_result(self.model.test_new(self.data.test_x), self.normal_params_filename)
				else:
					res = self.model.test_new(self.data.test_x)
				return [res]
			except:
				print("Error: data must be loaded before model can be used.")
				raise
				sys.exit()
				
		### MULTINET ###
		else:
			final_preds = []
			for i in range(0,self.num_builds):
				predlist = []
				for j in range(0,self.num_nodes):
					self.model_load_filename = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),"final_net_%d"%(j+1)))
					self.model = ecnet_model.multilayer_perceptron()
					self.model.load_net(self.model_load_filename)
					try:
						if self.normals_use == True:
							pred = ecnet_data_utils.denormalize_result(self.model.test_new(self.data.test_x), self.normal_params_filename)
						else:
							pred = self.model.test_new(self.data.test_x)
						predlist.append(pred)
					except:
						print("Error: data must be loaded before model can be used.")
						raise
						sys.exit()
				finalpred = []
				for j in range(0,len(predlist[0])):
					local_raw = []
					for k in range(0,len(predlist)):
						local_raw.append(predlist[k][j])
					finalpred.append([np.mean(local_raw)])
				final_preds.append(finalpred)
			return final_preds

	### Tests the model's RMSE on the currently loaded data set	(in its entirety)		
	def test_model_rmse(self):
		temp_test_x = self.data.test_x[:]
		temp_test_y = self.data.test_y[:]
		self.data.test_x = self.data.x
		self.data.test_y = self.data.y
			
		### SINGLE NET ###
		if self.folder_structs_built == False:
			self.model = ecnet_model.multilayer_perceptron()
			preds = self.use_mlp_model()
			rmse = ecnet_data_utils.calc_rmse(preds, self.data.test_y)
			self.data.test_x = temp_test_x
			self.data.test_y = temp_test_y
			return rmse
					
		### MULTINET ###
		else:
			self.model = ecnet_model.multilayer_perceptron()
			final_preds = self.use_mlp_model()
			rmse_list = []
			for i in range(0,len(final_preds)):
				rmse_list.append(ecnet_data_utils.calc_rmse(final_preds[i], self.data.test_y))
			self.data.test_x = temp_test_x
			self.data.test_y = temp_test_y
			return rmse_list
			
	def output_results(self, results, filename):
		ecnet_data_utils.output_results(results, self.data, filename)

	### Resaves the file under 'model_load_filename' to a new filename
	def resave_net(self, output):
		self.model = ecnet_model.multilayer_perceptron()
		try:
			self.model.load_net(self.model_load_filename)
		except:
			print("Error: Unable to load model; one must be created first. If one exists, check 'condig.yaml' for filename mismatches.")
			raise
			sys.exit()
		try:
			self.model.save_net(output)
		except:
			print("Error: Unable to save model. Unknown Error.")
			raise
			sys.exit()

# Creates the default folder structure, outlined in the file config by number of builds and nodes.
def create_folder_structure(server_obj):
	filename_raw = server_obj.model_output_filename
	server_obj.build_dirs = []
	for build_dirs in range(0,server_obj.num_builds):
		server_obj.build_dirs.append('build_%d'%(build_dirs + 1))
	server_obj.node_dirs = []
	for build_dirs in range(0,server_obj.num_builds):
		local_nodes = []
		for node_dirs in range(0,server_obj.num_nodes):
			local_nodes.append('node_%d'%(node_dirs + 1))
		server_obj.node_dirs.append(local_nodes)
	for build in range(0,len(server_obj.build_dirs)):
		path = os.path.join(server_obj.project_name, server_obj.build_dirs[build])
		if not os.path.exists(path):
			os.makedirs(path)
		for node in range(0,len(server_obj.node_dirs[build])):
			node_path = os.path.join(path, server_obj.node_dirs[build][node])
			if not os.path.exists(node_path):
				os.makedirs(node_path)
	server_obj.folder_structs_built = True

# Creates a model using config.yaml		
def create_model(server_obj):
	net = ecnet_model.multilayer_perceptron()
	net.addLayer(len(server_obj.data.x[0]), server_obj.mlp_in_layer_activ)
	for hidden in range(0,len(server_obj.mlp_hidden_layers)):
		net.addLayer(server_obj.mlp_hidden_layers[hidden][0], server_obj.mlp_hidden_layers[hidden][1])
	net.addLayer(len(server_obj.data.y[0]), server_obj.mlp_out_layer_activ)
	net.connectLayers()
	return net

# Imports 'config.yaml'; creates a default file if none is found	
def import_config():
	try:
		stream = open('config.yml', 'r')
		return(yaml.load(stream))
	except:
		print("Caught exception: config not found. Creating default config file.")
		create_default_config()
		stream = open('config.yml', 'r')
		return(yaml.load(stream))

# Creates a default 'config.yaml' file				
def create_default_config():
	stream = open('config.yml', 'w')
	config_dict = {
		'data_filename' : 'data.csv',
		'data_split' : [0.65,0.25,0.10],
		'learning_rate' : 0.1,
		'mlp_hidden_layers' : [[5, 'relu'], [5, 'relu']],
		'mlp_in_layer_activ' : 'relu',
		'mlp_out_layer_activ' : 'linear',
		'model_load_filename' : '_net_output',
		'model_output_filename' : '_net_output',
		'normal_params_filename' : 'normal_params',
		'normals_use' : False,
		'num_builds' : 1,
		'num_nodes' : 1,
		'num_trials' : 1,
		'param_limit_num' : 15,
		'project_name' : 'my_project',
		'sort_type' : 'random',	
		'train_epochs' : 100,
		'valid_max_epochs': 1000,
		'valid_mdrmse_stop' : 0.1,
		'valid_mdrmse_memory' : 1000
	}
	yaml.dump(config_dict,stream)
	
	
