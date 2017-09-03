"""
EXAMPLE SCRIPT:
Project creation and publishing

Creates a machine learning project using Server, creates the project save environment,
imports the training dataset, creates and fits models, selects the best models for each
build node, and publishes the project to a '.project' file
"""

from ecnet.server import Server

sv = Server()				         # Create the Server
sv.create_save_env()			         # Create project save environment

sv.import_data()			         # Import data from sv.vars['data_filename']
sv.fit_mlp_model_validation('shuffle_lv') 	 # Fits models for each node in each build, shuffling learn and validate sets
sv.select_best()			         # Selects the best performing model for each build's node

sv.publish_project()			         # Saves the project environment to a '.project' file
