"""
EXAMPLE SCRIPT:
Reduces database input parameter dimensionality

Imports a dataset, and reduces the number of input parameters to a specified number based
on input parameter performance
"""

from ecnet.server import Server

sv = Server()					                # Create the Server
sv.import_data()				                # Imports database
sv.limit_parameters(15, 'limited_database.csv')		# Limits input dimensionality to 15
