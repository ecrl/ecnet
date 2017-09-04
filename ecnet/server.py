#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet/server.py
#  v.1.2.3.dev1
#  Developed in 2017 by Travis Kessler <Travis_Kessler@student.uml.edu>
#
#  This program contains all the necessary config parameters and network serving functions
#

# 3rd party packages (open src.)
import yaml
import pickle
import numpy as np
import sys
import os
import zipfile
from shutil import copyfile

# ECNet program files
import ecnet.data_utils
import ecnet.model
import ecnet.limit_parameters
import ecnet.error_utils

### Config/server object; to be referenced by most other files ###
class Server:
    ### Initial declaration, handles config import
	def __init__(self):
		self.vars = {}
		self.vars.update(import_config(filename = 'config.yml'))
		self.folder_structs_built = False

	### Imports the data and stores it in the server	
	def import_data(self, data_filename = None):
		if data_filename is None:
			data_filename = self.vars['data_filename']
		try:
			self.data = ecnet.data_utils.initialize_data(data_filename)
		except:
			raise
			sys.exit()
		try:
			self.data.build()
		except:
			raise
			sys.exit()
		try:
			self.data.buildTVL(self.vars['data_sort_type'], self.vars['data_split'])
			if self.vars['normals_use'] == True:
				self.data.normalize('./tmp/normal_params')
			self.data.applyTVL()
			self.data.package()
		except:
			raise
			sys.exit()
	
	### Determines which 'param_num' parameters contribute to an accurate output; supply the number of parameters to limit to, and the output filename
	def limit_parameters(self, param_num, limited_database_output_filename):
		params = ecnet.limit_parameters.limit(self, param_num)
		ecnet.limit_parameters.output(self, params, limited_database_output_filename)

	### Creates the save environment
	def create_save_env(self):
		try:
			create_folder_structure(self)
		except:
			raise
			sys.exit()
	
	### Fits the model(s) using predetermined number of learning epochs	
	def fit_mlp_model(self, *args):
        ### PROJECT ###
		if self.folder_structs_built == True:
			self.fit_rmse_list = []
			for build in range(0,self.vars['project_num_builds']):
				build_rmse_list = []
				if self.vars['project_print_feedback'] == True:
					print("Build %d of %d"%(build+1,self.vars['project_num_builds']))
				for node in range(0,self.vars['project_num_nodes']):
					node_rmse_list = []
					if self.vars['project_print_feedback'] == True:
						print("Node %d of %d"%(node+1,self.vars['project_num_nodes']))
					for trial in range(0,self.vars['project_num_trials']):
						if self.vars['project_print_feedback'] == True:
							print("Trial %d of %d"%(trial+1,self.vars['project_num_trials']))
						self.output_filepath = os.path.join(os.path.join(os.path.join(self.vars['project_name'], self.build_dirs[build]), self.node_dirs[build][node]), "model_output" + "_%d"%(trial + 1))
						self.model = create_model(self)
						try:
							self.model.fit(self.data.learn_x, self.data.learn_y, self.vars['learning_rate'], self.vars['train_epochs'])
						except:
							raise
							sys.exit()
						try:
							res = self.model.test_new(self.data.x)
							rmse = ecnet.error_utils.calc_rmse(res, self.data.y)
							node_rmse_list.append(rmse)
						except:
							raise
							sys.exit()
						try:
							self.model.save_net(self.output_filepath)
						except:
							raise
							sys.exit()
						if 'shuffle_lv' in args:
							self.data.shuffle('l', 'v', data_split = self.vars['data_split'])
						elif 'shuffle_lvt' in args:
							self.data.shuffle('l', 'v', 't', data_split = self.vars['data_split'])
					build_rmse_list.append(node_rmse_list)
				self.fit_rmse_list.append(build_rmse_list)
		### SINGLE NET ###
		else:
			self.model = create_model(self)
			try:
				self.model.fit(self.data.learn_x, self.data.learn_y, self.vars['learning_rate'], self.vars['train_epochs'])
			except:
				raise
				sys.exit()
			try:
				self.model.save_net("./tmp/model_output")
			except:
				raise
				sys.exit()

	### Fits the model(s) using validation RMSE cutoff method, or max epochs
	def fit_mlp_model_validation(self, *args):
		### PROJECT ###
		if self.folder_structs_built == True:
			self.fit_rmse_list = []
			for build in range(0,self.vars['project_num_builds']):
				build_rmse_list = []
				if self.vars['project_print_feedback'] == True:
					print("Build %d of %d"%(build+1,self.vars['project_num_builds']))
				for node in range(0,self.vars['project_num_nodes']):
					node_rmse_list = []
					if self.vars['project_print_feedback'] == True:
						print("Node %d of %d"%(node+1,self.vars['project_num_nodes']))
					for trial in range(0,self.vars['project_num_trials']):
						if self.vars['project_print_feedback'] == True:
							print("Trial %d of %d"%(trial+1,self.vars['project_num_trials']))
						self.output_filepath = os.path.join(os.path.join(os.path.join(self.vars['project_name'], self.build_dirs[build]), self.node_dirs[build][node]), "model_output" + "_%d"%(trial + 1))
						self.model = create_model(self)
						try:
							self.model.fit_validation(self.data.learn_x, self.data.valid_x, self.data.learn_y, self.data.valid_y, self.vars['learning_rate'], self.vars['valid_mdrmse_stop'], self.vars['valid_mdrmse_memory'], self.vars['valid_max_epochs'])
						except:
							raise
							sys.exit()
						try:
							res = self.model.test_new(self.data.x)
							rmse = ecnet.error_utils.calc_rmse(res, self.data.y)
							node_rmse_list.append(rmse)
						except:
							raise
							sys.exit()
						try:
							self.model.save_net(self.output_filepath)
						except:
							raise
							sys.exit()
						if 'shuffle_lv' in args:
							self.data.shuffle('l', 'v', data_split = self.vars['data_split'])
						elif 'shuffle_lvt' in args:
							self.data.shuffle('l', 'v', 't', data_split = self.vars['data_split'])
					build_rmse_list.append(node_rmse_list)
				self.fit_rmse_list.append(build_rmse_list)
		### SINGLE NET ###
		else:
			self.model = create_model(self)
			try:
				self.model.fit_validation(self.data.learn_x, self.data.valid_x, self.data.learn_y, self.data.valid_y, self.vars['learning_rate'], self.vars['valid_mdrmse_stop'], self.vars['valid_mdrmse_memory'], self.vars['valid_max_epochs'])
			except:
				raise
				sys.exit()
			try:
				self.model.save_net("./tmp/model_output")
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
			# if you fit the model in the same script (self.fit_rmse_list is made)
			try:
				for build in range(len(self.fit_rmse_list)):
					for node in range(len(self.fit_rmse_list[build])):
						min_idx = 0
						for error in range(len(self.fit_rmse_list[build][node])):
							if self.fit_rmse_list[build][node][error] < self.fit_rmse_list[build][node][min_idx]:
								min_idx = error
						self.model_load_filename = os.path.join(os.path.join(self.vars['project_name'], "build_%d"%(build+1)), os.path.join("node_%d"%(node+1), "model_output" + "_%d"%(min_idx+1)))
						self.output_filepath = os.path.join(os.path.join(self.vars['project_name'], "build_%d"%(build+1)),os.path.join("node_%d"%(node+1),"final_net_%d"%(node+1)))
						self.resave_net
			# if you did not fit the model in the same script (self.fit_rmse_list is not made)
			except:
				for i in range(0,self.vars['project_num_builds']):
					for j in range(0,self.vars['project_num_nodes']):
						rmse_list = []
						for k in range(0,self.vars['project_num_trials']):
							self.model_load_filename = os.path.join(os.path.join(self.vars['project_name'], "build_%d"%(i+1)),os.path.join("node_%d"%(j+1), "model_output" + "_%d"%(k+1)))
							self.model = ecnet.model.multilayer_perceptron()
							self.model.load_net(self.model_load_filename)
							res = self.model.test_new(self.data.x)
							rmse = ecnet.error_utils.calc_rmse(res, self.data.y)
							rmse_list.append(rmse)
						current_min = 0
						for error in range(0,len(rmse_list)):
							if rmse_list[error] < rmse_list[current_min]:
								current_min = error
						self.model_load_filename = os.path.join(os.path.join(self.vars['project_name'], "build_%d"%(i+1)),os.path.join("node_%d"%(j+1), "model_output" + "_%d"%(current_min+1)))
						self.output_filepath = os.path.join(os.path.join(self.vars['project_name'], "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),"final_net_%d"%(j+1)))
						self.resave_net(self.output_filepath)
	
	### Predicts values for the current test set data
	def use_mlp_model(self, dset = None):
		x_vals = determine_x_vals(self, dset)
		### SINGLE MODEL ###
		if self.folder_structs_built == False:
			self.model = ecnet.model.multilayer_perceptron()
			try:
				self.model.load_net("tmp/model_output")
			except:
				raise
				sys.exit()
			try:
				if self.vars['normals_use'] == True:
					res = ecnet.data_utils.denormalize_result(self.model.test_new(x_vals), './tmp/normal_params')
				else:
					res = self.model.test_new(x_vals)
				return [res]
			except:
				raise
				sys.exit()
				
		### PROJECT ###
		else:
			final_preds = []
			# For each build
			for i in range(0,self.vars['project_num_builds']):
				predlist = []
				# For each node
				for j in range(0,self.vars['project_num_nodes']):
					self.model_load_filename = os.path.join(os.path.join(self.vars['project_name'], "build_%d"%(i+1)),os.path.join("node_%d"%(j+1),"final_net_%d"%(j+1)))
					self.model = ecnet.model.multilayer_perceptron()
					self.model.load_net(self.model_load_filename)
					try:
						if self.vars['normals_use'] == True:
							res = ecnet.data_utils.denormalize_result(self.model.test_new(x_vals), './tmp/normal_params')
						else:
							res = self.model.test_new(x_vals)
						predlist.append(res)
					except:
						raise
						sys.exit()
				finalpred = []
				# Check for one, or multiple outputs
				if self.data.controls_num_outputs is 1:
					for j in range(0,len(predlist[0])):
						local_raw = []
						for k in range(len(predlist)):
							local_raw.append(predlist[k][j])
						finalpred.append([np.mean(local_raw)])
					final_preds.append(finalpred)
				else:
					for j in range(len(predlist[0])):
						build_ave = []
						for k in range(len(predlist)):
							build_ave.append(predlist[k][j])
						finalpred.append(sum(build_ave)/len(build_ave))
					final_preds.append(list(finalpred))
			return final_preds
	
	### Calculates errors for each given argument		
	def calc_error(self, *args, dset = None):
		error_dict = {}
		preds = self.use_mlp_model(dset)
		y_vals = determine_y_vals(self, dset)
		for arg in args:
			### Tests the model's RMSE on the currently loaded data set	(in its entirety)
			if arg is 'rmse':
				### SINGLE MODEL ###
				if self.folder_structs_built == False:
					if self.vars['normals_use'] == True:
						rmse = ecnet.error_utils.calc_rmse(preds, ecnet.data_utils.denormalize_result(yvals, './tmp/normal_params'))
					else:
						rmse = ecnet.error_utils.calc_rmse(preds, y_vals)
					error_dict['RMSE'] = rmse		
				### PROJECT ###
				else:
					rmse_list = []
					for i in range(0,len(preds)):
						if self.vars['normals_use'] == True:
							rmse_list.append(ecnet.error_utils.calc_rmse(preds[i], ecnet.data_utils.denormalize_result(y_vals, './tmp/normal_params')))
						else:
							rmse_list.append(ecnet.error_utils.calc_rmse(preds[i], y_vals))
					error_dict['RMSE'] = rmse_list
			### Tests the model's coefficient of determination, or r-squared value
			elif arg is 'r2':
				### SINGLE MODEL ###
				if self.folder_structs_built == False:
					if self.vars['normals_use'] == True:
						r2 = ecnet.error_utils.calc_r2(preds, ecnet.data_utils.denormalize_result(y_vals, './tmp/normal_params'))
					else:
						r2 = ecnet.error_utils.calc_r2(preds, y_vals)
					error_dict['R-Squared'] = r2
				### PROJECT ###
				else:
					r2_list = []
					for i in range(0,len(preds)):
						if self.vars['normals_use'] == True:
							r2_list.append(ecnet.error_utils.calc_r2(preds[i], ecnet.data_utils.denormalize_result(y_vals, './tmp/normal_params')))
						else:
							r2_list.append(ecnet.error_utils.calc_r2(preds[i], y_vals))
					error_dict['R-Squared'] = r2_list
			### Tests the model's mean absolute error on the currently loaded data set (in its entirety)
			elif arg is 'mean_abs_error':
				### SINGLE MODEL ###
				if self.folder_structs_built == False:
					if self.vars['normals_use'] == True:
						mae = ecnet.error_utils.calc_mean_abs_error(preds, ecnet.data_utils.denormalize_result(y_vals, './tmp/normal_params'))
					else:
						mae = ecnet.error_utils.calc_mean_abs_error(preds, self.data.y)
					error_dict['Mean Average Error'] = mae
				### PROJECT ###
				else:
					mae_list = []
					for i in range(0,len(preds)):
						if self.vars['normals_use'] == True:
							mae_list.append(ecnet.error_utils.calc_mean_abs_error(preds[i], ecnet.data_utils.denormalize_result(y_vals, './tmp/normal_params')))
						else:
							mae_list.append(ecnet.error_utils.calc_mean_abs_error(preds[i], y_vals))
					error_dict['Mean Average Error'] = mae_list
			### Tests the model's median absolute error on the currently loaded data set (in its entirety)
			elif arg is 'med_abs_error':
				### SINGLE MODEL ###
				if self.folder_structs_built == False:
					if self.vars['normals_use'] == True:
						medae = ecnet.error_utils.calc_med_abs_error(preds, ecnet.data_utils.denormalize_result(y_vals, './tmp/normal_params'))
					else:
						medae = ecnet.error_utils.calc_med_abs_error(preds, y_vals)
					error_dict['Median Absolute Error'] = medae
				### PROJECT ###
				else:
					medae_list = []
					for i in range(0,len(preds)):
						if self.vars['normals_use'] == True:
							medae_list.append(ecnet.error_utils.calc_med_abs_error(preds[i], ecnet.data_utils.denormalize_result(y_vals, './tmp/normal_params')))
						else:
							medae_list.append(ecnet.error_utils.calc_med_abs_error(preds[i], y_vals))
					error_dict['Median Absolute Error'] = medae_list
			else:
				print("Error: unknown/unsupported error function - ",arg)
				return
		return error_dict
		
	### Outputs results to desired .csv file	
	def output_results(self, results, filename):
		ecnet.data_utils.output_results(results, self.data, filename)

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
			
	### Cleans up the project directory (only keep final node NN's), copies the config, data and normal params (if present) files to the directory, and zips the directory for publication
	def publish_project(self):
		# Clean up project directory
		for build in range(self.vars['project_num_builds']):
			for node in range(self.vars['project_num_nodes']):
				directory = os.path.join(self.vars['project_name'], os.path.join('build_%d'%(build+1), 'node_%d'%(node+1)))
				filelist = [f for f in os.listdir(directory) if 'model_output' in f]
				for f in filelist:
					os.remove(os.path.join(directory, f))
		# Copy config.yml and normal parameters file to the project directory
		save_config(self.vars)
		copyfile('config.yml', os.path.join(self.vars['project_name'], 'config.yml'))
		if self.vars['normals_use'] is True:
			copyfile('./tmp/normal_params.ecnet', os.path.join(self.vars['project_name'], 'normal_params.ecnet'))
		# Export the currently loaded dataset
		try:
			pickle.dump(self.data, open(os.path.join(self.vars['project_name'],'data.d'),'wb'))
		except:
			pass
		# Zip up the project
		zipf = zipfile.ZipFile(self.vars['project_name'] + '.project', 'w', zipfile.ZIP_DEFLATED)
		for root, dirs, files in os.walk(self.vars['project_name']):
			for file in files:
				zipf.write(os.path.join(root,file))
		zipf.close()
	
	### Opens a published project, importing model, data, config, normal params
	def open_project(self, project_name):
		# Check naming scheme
		if '.project' not in project_name:
			zip_loc = project_name + '.project'
		else:
			project_name = project_name.replace('.project', '')
			zip_loc = project_name + '.project'
		# Unzip project to directory
		zip_ref = zipfile.ZipFile(zip_loc, 'r')
		zip_ref.extractall('./')
		zip_ref.close()
		# Update config to project config
		self.vars.update(import_config(filename = os.path.join(project_name,'config.yml')))
		save_config(self.vars)
		# Unpack data
		try:
			self.data = pickle.load(open(os.path.join(project_name, 'data.d'),'rb'))
		except:
			print('Error: unable to load data.')
			pass
		# Set up model environment
		self.folder_structs_built = True
		create_folder_structure(self)
		self.model = create_model(self)

# Creates the default folder structure, outlined in the file config by number of builds and nodes.
def create_folder_structure(server_obj):
	server_obj.build_dirs = []
	for build_dirs in range(0,server_obj.vars['project_num_builds']):
		server_obj.build_dirs.append('build_%d'%(build_dirs + 1))
	server_obj.node_dirs = []
	for build_dirs in range(0,server_obj.vars['project_num_builds']):
		local_nodes = []
		for node_dirs in range(0,server_obj.vars['project_num_nodes']):
			local_nodes.append('node_%d'%(node_dirs + 1))
		server_obj.node_dirs.append(local_nodes)
	for build in range(0,len(server_obj.build_dirs)):
		path = os.path.join(server_obj.vars['project_name'], server_obj.build_dirs[build])
		if not os.path.exists(path):
			os.makedirs(path)
		for node in range(0,len(server_obj.node_dirs[build])):
			node_path = os.path.join(path, server_obj.node_dirs[build][node])
			if not os.path.exists(node_path):
				os.makedirs(node_path)
	server_obj.folder_structs_built = True

# Used by use_mlp_model to determine which x-values to use for calculations
def determine_x_vals(server, dset):
	if dset is 'test':
		x_vals = server.data.test_x
	elif dset is 'learn':
		x_vals = server.data.learn_x
	elif dset is 'valid':
		x_vals = server.data.valid_x
	elif dset is 'train':
		x_vals = []
		for i in range(len(server.data.learn_x)):
			x_vals.append(list(server.data.learn_x[i]))
		for i in range(len(server.data.valid_x)):
			x_vals.append(list(server.data.valid_x[i]))
	else:
		x_vals = server.data.x
	return x_vals

# Used by calc_error to determine which y-values to use for error calculations
def determine_y_vals(server, dset):
	if dset is 'test':
		y_vals = server.data.test_y
	elif dset is 'learn':
		y_vals = server.data.learn_y
	elif dset is 'valid':
		y_vals = server.data.valid_y
	elif dset is 'train':
		y_vals = []
		for i in range(len(server.data.learn_y)):
			y_vals.append(list(server.data.learn_y[i]))
		for i in range(len(server.data.valid_y)):
			y_vals.append(list(server.data.valid_y[i]))
	else:
		y_vals = server.data.y
	return y_vals

# Creates a model using config.yaml		
def create_model(server_obj):
	net = ecnet.model.multilayer_perceptron()
	net.addLayer(len(server_obj.data.x[0]), server_obj.vars['mlp_in_layer_activ'])
	for hidden in range(0,len(server_obj.vars['mlp_hidden_layers'])):
		net.addLayer(server_obj.vars['mlp_hidden_layers'][hidden][0], server_obj.vars['mlp_hidden_layers'][hidden][1])
	net.addLayer(len(server_obj.data.y[0]), server_obj.vars['mlp_out_layer_activ'])
	net.connectLayers()
	return net

# Imports 'config.yaml'; creates a default file if none is found	
def import_config(filename = 'config.yml'):
	try:
		stream = open(filename, 'r')
		return(yaml.load(stream))
	except:
		create_default_config()
		stream = open(filename, 'r')
		return(yaml.load(stream))

# Saves all server variables to config.yml
def save_config(config_dict):
	with open('config.yml', 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style = False, explicit_start = True)

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
		'normals_use' : False,
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
	
	
