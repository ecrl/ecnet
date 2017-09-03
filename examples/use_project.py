"""
EXAMPLE SCRIPT:
Using a pre-existing project to obtain results

Imports a pre-existing project to the Server environment, imports a testing
dataset, obtain results and errors for testing dataset
"""

from ecnet.server import Server

sv = Server()					    # Create the Server
sv.open_project('cn_v1.0_project')	            # Opens pre-existing project

sv.import_data('testing_data.csv')				    # Import a new set to test

results = sv.use_mlp_model()				            # Grab results for whole test set
sv.output_results(results, 'testing_results.csv')                  # Save results to file

errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error')	 # Compute errors
print(errors)	                                                         # List errors
