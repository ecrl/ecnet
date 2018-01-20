"""
EXAMPLE SCRIPT:
Creates a static test

Imports a dataset, and creates two files; one containing the test data,
	one containing the training (learning + validation) data. Set sizes
	are determined by 'data_split' server variable.
"""

from ecnet.server import Server
from ecnet import data_utils

# Create the Server
sv = Server()

# [learn%, validation%, test%] *** 10% of data will be used for a static test set
sv.vars['data_split'] = [0.7, 0.2, 0.1]

# Imports database
sv.import_data('original_database.csv')

# Creates a static test set using randomly imported test proportion
#	output files will be:
#		original_database_slv.csv
#		original_database_st.csv
data_utils.create_static_test_set(sv.data)