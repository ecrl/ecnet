"""
EXAMPLE SCRIPT:
Select best trial from each node using static test set performance

This script operates as follows:
	Train models using training (learning + validation) set
	Import test set, select best trials using test set performance
	Obtain results and errors for test set
	Obtain results and errors for training set
	Publish project
"""

from ecnet.server import Server

# create server object
sv = Server()
# create the project folder structure
sv.create_save_env()

# set data split ([learn, validate, test]) and import the training set
sv.vars['data_split'] = [0.7, 0.3, 0.0]
sv.import_data('slv_data.csv')

# fit the model to the training set, shuffling learn and validate sets between trials
sv.fit_mlp_model_validation('shuffle_lv')

# import the test set, select best trials based on test set performance
sv.import_data('st_data.csv')
sv.select_best()

# obtain predictions for test set
test_results = sv.use_mlp_model()
sv.output_results(test_results, filename = 'test_set_results.csv')

# obtain and print errors for test set
test_errors = sv.calc_error('rmse', 'r2', 'mean_abs_error', 'med_abs_error')
print()
print('Test Errors:')
print(test_errors)
print()

# obtain predictions for training set
sv.import_data('slv_data.csv')
train_results = sv.use_mlp_model()
sv.output_results(train_results, filename = 'training_set_results.csv')

# obtain and print errors for training set
train_errors = sv.calc_error('rmse', 'r2', 'mean_abs_error', 'med_abs_error')
print()
print('Training Errors:')
print(train_errors)
print()

# publish (i.e. save the state) of current project
sv.publish_project()