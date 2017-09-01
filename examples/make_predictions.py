"""
EXAMPLE SCRIPT
Creating the Server object (for handling project directories, machine learning models, and data handling),
creating the folder hierarchy for your project (project name, number of build sets, and nodes/set, specified in config),
importing data from database (specified in config), fitting X machine learning models (where X is number of trials/node,
specified in config), selecting the best performing trial from each node for the final build set, testing for multiple 
errors on your database, predicting values for the database and outputting them to an output file.
""" 
from ecnet.server import Server

sv = Server()							# Create server object
sv.create_save_env()					# Create a folder structure for your project
sv.import_data()						# Import data
sv.create_mlp_model()					# Create a multilayer perceptron (neural network)
sv.fit_mlp_model_validation()			# Fits the mlp using input database (w/ validation set)
sv.select_best()						# Select best trial from each build node

results = sv.use_mlp_model_all()					# Calculate results from data (all sets)
sv.output_results(results, "all", "cn_results.csv")	# Output results to specified file

errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error')	# Calculates errors for dataset predictions
print(errors)
