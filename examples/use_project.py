"""
EXAMPLE SCRIPT:
Using a pre-existing project to obtain results

Imports a pre-existing project to the Server environment, imports a testing
dataset, obtain results and errors for testing dataset
"""

from ecnet.server import Server


# Create the Server
sv = Server()

# Opens pre-existing project
sv.open_project('cn_v1.0_project')

# Import a new set to predict values for
sv.import_data('testing_data.csv')

# Grab results (for whole set)
results = sv.use_mlp_model()

# Save results to file
sv.output_results(results, filename='testing_results.csv')

# Compute and print errors (for whole set)
errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error')
print(errors)
