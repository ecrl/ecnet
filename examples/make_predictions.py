"""
EXAMPLE SCRIPT
Creating the Server object (for handling project directories, machine learning models, and data handling),
creating the folder hierarchy for your project (project name, number of build sets, and nodes/set, specified in config),
importing data from database (specified in config), fitting X machine learning models (where X is number of trials/node,
specified in config), selecting the best performing trial from each node for the final build set, testing for multiple 
errors, predicting values for the database and outputting them to an output file.
"""
import ecnet

server = ecnet.server.Server()			# Create server object
server.create_save_env()				# Create a folder structure for your project
server.import_data()					# Import data
server.create_mlp_model()				# Create a multilayer perceptron (neural network)
server.fit_mlp_model_validation()		# Fits the mlp using the input database (with a validation set)
server.select_best()					# Select best trial from each build node

rmse = server.test_model_rmse()			# Root-mean-square error
mae = server.test_model_mae()			# Mean average error
r2 = server.test_model_r2()				# Coeffecient of determination (r-squared)

results = server.use_mlp_model_all()	# Calculate results from data (all values, not just test set)
server.output_results(results, "all", "results_output.csv")		# Output results to specified file

print(rmse)
print(mae)
print(r2)
