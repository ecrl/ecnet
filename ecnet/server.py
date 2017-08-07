#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet_server.py
#  
#  Developed in 2017 by Travis Kessler <Travis_Kessler@student.uml.edu>
#
#  This program contains all the necessary config parameters and network serving functions
#

# 3rd party packages (open src.)
import yaml
import numpy as np
import sys
import os

# ECNet program files
import ecnet.data_utils
import ecnet.model
import ecnet.limit_parameters

### Config/server object; to be referenced by most other files ###
class Server:
    ### Initial declaration, handles config import
	def __init__(self):
		self.__dict__.update(import_config())
		self.folder_structs_built = False

	### Imports the data and stores it in the server	
	def import_data(self):
		try:
			self.data = ecnet.data_utils.initialize_data(self.data_filename)
		except:
			raise
			sys.exit()
		try:
			self.data.build()
		except:
			raise
			sys.exit()
		try:
			self.data.buildTVL(self.data_sort_type, self.data_split)
			if self.normals_use == True:
				self.data.normalize(self.normal_params_filename)
			self.data.applyTVL()
			self.data.package()
		except:
			raise
			sys.exit()
	
	### Determines which parameters contribute to an accurate output - number is determined by param_limit_num in config; supply an output filename
	def limit_parameters(self, limited_database_output_filename):
		params = ecnet.limit_parameters.limit(self)
		ecnet.limit_parameters.output(self, params, limited_database_output_filename)

	### Creates the save environment
	def create_save_env(self):
		try:
			create_folder_structure(self)
		except:
			raise
			sys.exit()

	### Creates the model once the data is loaded		
	def create_mlp_model(self):
		try:
			self.model = create_model(self)
		except:
			raise
			sys.exit()
	
	### Fits the model using predetermined number of learning epochs	
	def fit_mlp_model(self):
        ### PROJECT ###
		if self.folder_structs_built == True:
			for build in range(0,self.project_num_builds):
				if self.project_print_feedback == True:
					print("Build %d of %d"%(build+1,self.project_num_builds))
				for node in range(0,self.project_num_nodes):
					if self.project_print_feedback == True:
						print("Node %d of %d"%(node+1,self.project_num_nodes))
					for trial in range(0,self.project_num_trials):
						if self.project_print_feedback == True:
							print("Trial %d of %d"%(trial+1,self.project_num_trials))
						self.output_filepath = os.path.join(os.path.join(os.path.join(self.project_name, self.build_dirs[build]), self.node_dirs[build][node]), "model_output" + "_%d"%(trial + 1))
						try:
							self.model.fit(self.data.learn_x, self.data.learn_y, self.learning_rate, self.train_epochs)
						except:
							raise
							sys.exit()
						try:
							self.model.save_net(self.output_filepath)
						except:
							raise
							sys.exit()
						self.import_data()
						self.create_mlp_model()
		### SINGLE NET ###
		else:
			try:
				self.model.fit(self.data.learn_x, self.data.learn_y, self.learning_rate, self.train_epochs)
			except:
				raise
				sys.exit()
			try:
				self.model.save_net("model_output")
			except:
				raise
				sys.exit()

	### Fits the model using validation RMSE cutoff method, or max epochs
	def fit_mlp_model_validation(self):
		### PROJECT ###
		if self.folder_structs_built == True:
			for build in range(0,self.project_num_builds):
				if self.project_print_feedback == True:
					print("Build %d of %d"%(build+1,self.project_num_builds))
				for node in range(0,self.project_num_nodes):
					if self.project_print_feedback == True:
						print("Node %d of %d"%(node+1,self.project_num_nodes))
					for trial in range(0,self.project_num_trials):
						if self.project_print_feedback == True:
							print("Trial %d of %d"%(trial+1,self.project_num_trials))
						self.output_filepath = os.path.join(os.path.join(os.path.join(self.project_name, self.build_dirs[build]), self.node_dirs[build][node]), "model_output" + "_%d"%(trial + 1))
						try:
							self.model.fit_validation(self.data.learn_x, self.data.valid_x, self.data.learn_y, self.data.valid_y, self.learning_rate, self.valid_mdrmse_stop, self.valid_mdrmse_memory, self.valid_max_epochs)
						except:
							raise
							sys.exit()
						try:
							self.model.save_net(self.output_filepath)
						except:
							raise
							sys.exit()
						self.import_data()
						self.create_mlp_model()
		### SINGLE NET ###
		else:
			try:
				self.model.fit_validation(self.data.learn_x, self.data.valid_x, self.data.learn_y, self.data.valid_y, self.learning_rate, self.valid_mdrmse_stop, self.valid_mdrmse_memory, self.valid_max_epochs)
			except:
				raise
				sys.exit()
			try:
				self.model.save_net("model_output")
			except:
				raise
				sys.exit()

	### Selects the best performing networks from each node of each build. Folder structs must be created.			
	def select_best(self):
		### SINGLE MODEL ###
		if self.folder_structs_built == False:
			print("Error: Project folder structure must be built in order to select best.")
			sys.exit()
		### PROJECT ###
		else:
			for i in range(0,self.project_num_builds):
				for j in range(0,self.project_num_nodes):
					rmse_list = []
					for k in range(0,self.project_num_trials):
						self.model_load_filename = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1), "model_output" + "_%d"%(k+1)))
						self.model = ecnet.model.multilayer_perceptron()
						self.model.load_net(self.model_load_filename)
						res = self.model.test_new(self.data.x)
						rmse = ecnet.data_utils.calc_rmse(res, self.data.y)
						rmse_list.append(rmse)
					current_min = 0
					for error in range(0,len(rmse_list)):
						if rmse_list[error] < rmse_list[current_min]:
							current_min = error
					self.model_load_filename = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1), "model_output" + "_%d"%(current_min+1)))
					self.output_filepath = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),"final_net_%d"%(j+1)))
					self.resave_net(self.output_filepath)
	
	### Predicts values for the current test set data
	def use_mlp_model(self):
		### SINGLE MODEL ###
		if self.folder_structs_built == False:
			self.model = ecnet.model.multilayer_perceptron()
			try:
				self.model.load_net("model_output")
			except:
				raise
				sys.exit()
			try:
				if self.normals_use == True:
					res = ecnet.data_utils.denormalize_result(self.model.test_new(self.data.test_x), self.normal_params_filename)
				else:
					res = self.model.test_new(self.data.test_x)
				return [res]
			except:
				raise
				sys.exit()
				
		### PROJECT ###
		else:
			final_preds = []
			for i in range(0,self.project_num_builds):
				predlist = []
				for j in range(0,self.project_num_nodes):
					self.model_load_filename = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),"final_net_%d"%(j+1)))
					self.model = ecnet.model.multilayer_perceptron()
					self.model.load_net(self.model_load_filename)
					try:
						if self.normals_use == True:
							pred = ecnet.data_utils.denormalize_result(self.model.test_new(self.data.test_x), self.normal_params_filename)
						else:
							pred = self.model.test_new(self.data.test_x)
						predlist.append(pred)
					except:
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
		
	### Predicts values for the current data set (whole)
	def use_mlp_model_all(self):
		### SINGLE MODEL ###
		if self.folder_structs_built == False:
			self.model = ecnet.model.multilayer_perceptron()
			try:
				self.model.load_net("model_output")
			except:
				raise
				sys.exit()
			try:
				if self.normals_use == True:
					res = ecnet.data_utils.denormalize_result(self.model.test_new(self.data.x), self.normal_params_filename)
				else:
					res = self.model.test_new(self.data.x)
				return [res]
			except:
				raise
				sys.exit()
				
		### PROJECT ###
		else:
			final_preds = []
			for i in range(0,self.project_num_builds):
				predlist = []
				for j in range(0,self.project_num_nodes):
					self.model_load_filename = os.path.join(os.path.join(self.project_name, "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),"final_net_%d"%(j+1)))
					self.model = ecnet.model.multilayer_perceptron()
					self.model.load_net(self.model_load_filename)
					try:
						if self.normals_use == True:
							pred = ecnet.data_utils.denormalize_result(self.model.test_new(self.data.x), self.normal_params_filename)
						else:
							pred = self.model.test_new(self.data.x)
						predlist.append(pred)
					except:
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
		### SINGLE MODEL ###
		if self.folder_structs_built == False:
			preds = self.use_mlp_model_all()
			if self.normals_use == True:
				rmse = ecnet.data_utils.calc_rmse(preds, ecnet.data_utils.denormalize_result(self.data.y, self.normal_params_filename))
			else:
				rmse = ecnet.data_utils.calc_rmse(preds, self.data.y)
			return rmse			
		### PROJECT ###
		else:
			final_preds = self.use_mlp_model_all()
			rmse_list = []
			for i in range(0,len(final_preds)):
				if self.normals_use == True:
					rmse_list.append(ecnet.data_utils.calc_rmse(final_preds[i], ecnet.data_utils.denormalize_result(self.data.y, self.normal_params_filename)))
				else:
					rmse_list.append(ecnet.data_utils.calc_rmse(final_preds[i], self.data.y))
			return rmse_list
			
	### Tests the model's mean absolute error on the currently loaded data set (in its entirety)
	def test_model_mae(self):
		### SINGLE MODEL ###
		if self.folder_structs_built == False:
			preds = self.use_mlp_model_all()
			if self.normals_use == True:
				mae = ecnet.data_utils.calc_mae(preds, ecnet.data_utils.denormalize_result(self.data.y, self.normal_params_filename))
			else:
				mae = ecnet.data_utils.calc_mae(preds, self.data.y)
			return mae	
		### PROJECT ###
		else:
			final_preds = self.use_mlp_model_all()
			mae_list = []
			for i in range(0,len(final_preds)):
				if self.normals_use == True:
					mae_list.append(ecnet.data_utils.calc_mae(final_preds[i], ecnet.data_utils.denormalize_result(self.data.y, self.normal_params_filename)))
				else:
					mae_list.append(ecnet.data_utils.calc_mae(final_preds[i], self.data.y))
			return mae_list
		
	### Tests the model's coefficient of determination, or r-squared value
	def test_model_r2(self):
		### SINGLE MODEL ###
		if self.folder_structs_built == False:
			preds = self.use_mlp_model_all()
			if self.normals_use == True:
				r2 = ecnet.data_utils.calc_r2(preds, ecnet.data_utils.denormalize_result(self.data.y, self.normal_params_filename))
			else:
				r2 = ecnet.data_utils.calc_r2(preds, self.data.y)
			return r2
		### PROJECT ###
		else:
			final_preds = self.use_mlp_model_all()
			r2_list = []
			for i in range(0,len(final_preds)):
				if self.normals_use == True:
					r2_list.append(ecnet.data_utils.calc_r2(final_preds[i], ecnet.data_utils.denormalize_result(self.data.y, self.normal_params_filename)))
				else:
					r2_list.append(ecnet.data_utils.calc_r2(final_preds[i], self.data.y))
			return r2_list
		
	### Outputs results to desired .csv file	
	def output_results(self, results, which_data, filename):
		ecnet.data_utils.output_results(results, self.data, which_data, filename)

	### Resaves the file under 'self.model_load_filename' to specified output filepath
	def resave_net(self, output):
		self.model = ecnet.model.multilayer_perceptron()
		try:
			self.model.load_net(self.model_load_filename)
		except:
			raise
			sys.exit()
		try:
			self.model.save_net(output)
		except:
			raise
			sys.exit()

# Creates the default folder structure, outlined in the file config by number of builds and nodes.
def create_folder_structure(server_obj):
	server_obj.build_dirs = []
	for build_dirs in range(0,server_obj.project_num_builds):
		server_obj.build_dirs.append('build_%d'%(build_dirs + 1))
	server_obj.node_dirs = []
	for build_dirs in range(0,server_obj.project_num_builds):
		local_nodes = []
		for node_dirs in range(0,server_obj.project_num_nodes):
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
	net = ecnet.model.multilayer_perceptron()
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
		'data_sort_type' : 'random',
		'data_split' : [0.65,0.25,0.10],
		'learning_rate' : 0.1,
		'mlp_hidden_layers' : [[5, 'relu'], [5, 'relu']],
		'mlp_in_layer_activ' : 'relu',
		'mlp_out_layer_activ' : 'linear',
		'normal_params_filename' : 'normal_params',
		'normals_use' : False,
		'param_limit_num' : 15,
		'project_name' : 'my_project',
		'project_num_builds' : 1,
		'project_num_nodes' : 1,
		'project_num_trials' : 1,
		'project_print_feedback': True,
		'train_epochs' : 100,
		'valid_max_epochs': 1000,
		'valid_mdrmse_stop' : 0.1,
		'valid_mdrmse_memory' : 1000
	}
	yaml.dump(config_dict,stream)
	
	
