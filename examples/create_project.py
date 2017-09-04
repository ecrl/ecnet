"""
EXAMPLE SCRIPT:
Project creation and publishing

Creates a machine learning project using Server, creates the project save environment,
imports the training dataset, creates and fits models, selects the best models for each
build node, and publishes the project to a '.project' file
"""

from ecnet.server import Server

# Create the Server
sv = Server()

# Create project save environment
sv.create_save_env()

# Import data from config. file database
sv.import_data()

# Fits models for each node in each build, shuffling learn and validate sets between trials
sv.fit_mlp_model_validation('shuffle_lv')

# Selects the best performing model from each build's node
sv.select_best()

# Saves the project environment to a '.project' file
sv.publish_project()
