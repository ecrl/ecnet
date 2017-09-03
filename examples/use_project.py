"""
EXAMPLE SCRIPT:
Using a pre-existing project to obtain results

Imports a pre-existing project to the Server environment, imports a testing
dataset, obtain results and errors for testing dataset
"""

from ecnet.server import Server

sv = Server()					    # Create the Server
sv.open_project('cn_v1.0_project')	            # Opens pre-existing project

sv.vars['data_filename'] = ('testing_data.csv')     # Change data filename to a test set
sv.import_data()				    # Import the test set

test_results = sv.use_mlp_model_all()				# Grab results for whole test set
sv.output_results(test_results, 'all', 'testing_results.csv')   # Save results to file

test_errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error')	 # Compute errors
print(test_errors)	                                                         # List errors
