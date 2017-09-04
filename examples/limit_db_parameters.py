"""
EXAMPLE SCRIPT:
Reduces database input parameter dimensionality

Imports a dataset, and reduces the number of input parameters to a specified number based
on input parameter performance
"""

from ecnet.server import Server

# Create the Server
sv = Server()

# Imports config. database
sv.import_data()

# Limits input dimensionality to 15, saves to specified file
sv.limit_parameters(15, 'limited_database.csv')
