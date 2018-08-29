# Import the Server object
from ecnet import Server

# Create Server
sv = Server()

# Import data (change 'my_data.csv' to your database name)
sv.import_data('my_data.csv')

# Limit the input dimensionality to 15, save to 'my_data_limited.csv'
sv.limit_parameters(15, 'my_data_limited.csv')


# Use this line instead for limiting the input dimensionality using a genetic
# algorithm
sv.limit_parameters(15, 'my_data_limited_genetic.csv', use_genetic=True)
