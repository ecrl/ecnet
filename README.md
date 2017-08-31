![UML Energy & Combustion Research Laboratory](http://faculty.uml.edu/Hunter_Mack/uploads/9/7/1/3/97138798/1481826668_2.png)

# ECNet - Large scale machine learning projects for fuel property prediction
### Developed by Travis Kessler with direction from Professor John Hunter Mack
	
**ECNet** is an open source Python package for large scale machine learning projects with a focus on fuel property prediction. A __project__ is considered a collection of __builds__, and each build is a collection of __nodes__. Nodes are averaged to obtain a final predicted value for the build. For each node in a build, multiple neural networks are constructed and the best performing neural network is used for that node's predicted value.

[T. Sennott et al.](https://www.researchgate.net/publication/267576682_Artificial_Neural_Network_for_Predicting_Cetane_Number_of_Biofuel_Candidates_Based_on_Molecular_Structure) have shown that artificial neural networks (ANN's) can be applied to cetane number prediction with relatively little error. ECNet provides scientists an open source tool for predicting key fuel properties of potential next-generation biofuels, reducing the need for costly fuel synthesis and experimentation.

Using ECNet, [T. Kessler et al.](https://www.researchgate.net/publication/317569746_Artificial_neural_network_based_predictions_of_cetane_number_for_furanic_biofuel_additives) have increased the generalizability of ANN's to predict the cetane numbers for a variety of molecular classes represented in the cetane number database, and have increased the accuracy of ANN's for predicting the cetane numbers of underrepresented molecular classes through targeted database expansion.

## Installation:

### Prerequisites:
- Have Python 3.5 installed
- Have the ability to install Python packages

### Method 1: pip
If you are working in a Linux/Mac environment:
- **sudo pip install ecnet**

Alternatively, in a Windows or virtualenv environment:
- **pip install ecnet**

Note: if multiple Python releases are installed on your system (e.g. 2.7 and 3.5), you may need to execute the correct version of pip. For Python 3.5, change **"pip install ecnet"** to **"pip3 install ecnet"**.

### Method 2: From source
- Download the ECNet repository, navigate to the download location on the command line/terminal, and execute 
**"python setup.py install"**. 

Additional package dependencies (TensorFlow, PyYaml) will be installed during the ECNet installation process.

## Usage:

ECNet operates using a **Server** object that interfaces with data utility classes and neural network creation classes. The Server object handles importing data and model creation for your project, and serves the data to the model. Configurable variables like your project's name, number of builds and nodes, ANN learning and architecture variables, data splitting controls, and more are found in a __config.yml__ file in your working directory. Here's what a config.yml file for cetane number prediction looks like:

```yml
data_filename: cn_model_v1.0.csv
data_sort_type: random
data_split: [0.65, 0.25, 0.1]
learning_rate: 0.05
mlp_hidden_layers:
- [32, relu]
- [32, relu]
mlp_in_layer_activ: relu
mlp_out_layer_activ: linear
normal_params_filename: normalization_parameters
normals_use: False
param_limit_num: 15
project_name: cn_v1.0_project
project_num_builds: 25
project_num_nodes: 5
project_num_trials: 75
project_print_feedback: True
train_epochs: 12500
valid_max_epochs: 7000
valid_mdrmse_stop: 0.0001
valid_mdrmse_memory: 1000
```

Here are brief explanations of each of these variables:
- **data_filename**: the location of your formatted .csv database for training and testing data
- **data_sort_type** - random or explicit: how the learning, validation and testing data sets should be chosen
	- Note: explicitly split requires set locations to be defined inside the database
- **data_split** - [learning, validation, testing]: proportions used for the random sort type
- **learning_rate**: value passed to the AdamOptimizer to use as its learning rate during training
- **mlp_hidden_layers** - [[num_neurons_0, layer_type_0],...,[num_neurons_n, layer_type_n]]: the architecture of the ANN between the input and output layers
	- Rectified linear unit ('relu') and Sigmoid ('sigmoid') layer types are currently supported
- **mlp_in_layer_activ** - the layer type of the input layer; number of nodes is determined by data dimensions
- **mlp_out_layer_activ** - the layer type of the output layer; number of nodes is determined by data dimensions
- **normal_params_filename** - filename for the normalization parameters (if used)
- **normals_use** - boolean to determine if parameters should be normalized (min-max, between 0 and 1)
- **param_limit_num** - used by the parameter limiting methods, determines how many optimal parameters to retain from a very large dimensional database
- **project_name** - the name of your project
- **project_num_builds** - the number of builds in your project
- **project_num_nodes** - the number of nodes in each build
- **project_num_trials** - the number of ANN's to be constructed in each node
- **project_print_feedback** - whether the console will show status messages
- **train_epochs** - number of training iterations (not used with validation)
- **valid_max_epochs** - the maximum number of training iterations during the validation process
- **valid_mdrmse_stop** - the threshold to determine learning cutoff (looks at the change in validation RMSE over time)
- **valid_mdrmse_memory** - how many epochs back the validation process looks in determining the change in validation RMSE over time

Now let's look at a script for building a project, predicting values of cetane number for molecules in the cetane number database, and saving the results to an output file: 

```python
from ecnet.server import Server

sv = Server()						# Create server object
sv.create_save_env()					# Create a folder structure for your project
sv.import_data()					# Import data
sv.create_mlp_model()					# Create a multilayer perceptron (neural network)
sv.fit_mlp_model_validation()				# Fits the mlp using input database (w/ validation set)
sv.select_best()					# Select best trial from each build node

results = sv.use_mlp_model_all()			# Calculate results from data (all sets)
sv.output_results(results, "all", "cn_results.csv")	# Output results to specified file

rmse = sv.test_model_rmse()				# Root-mean-square error
mae = sv.test_model_mae()				# Mean average error
r2 = sv.test_model_r2()					# Coeffecient of determination (r-squared)

print(rmse)
print(mae)
print(r2)

```

You can change all the configuration variables from your Python script, without having to edit and reopen your config.yml file. For example, changing the filename for your input database looks like this:

```python
from ecnet.server import Server

sv = Server()
sv.data_filename = 'my_database.csv'
```

Here's an overview of the Server object's methods:

- **create_save_env()**: creates the folder hierarchy for your project, contained in a folder named after your project name
- **import_data()**: imports the data from the database specified in 'data_filename', splits the data into learning/validation/testing groups, and packages the data so it's ready to be sent to the model
- **create_mlp_model()**: creates a multilayer-perceptron (aka, a feed-forward neural network) based on the input/output dimensionality of the imported data and the hidden layer architecture specified in the config
- **fit_mlp_model()**: fits the multilayer-perceptron to the data, for 'train_epochs' learning iterations; no validation done here
- **fit_mlp_model_validation()**: fits the multilayer-perceptron to the data, using the validation set to determine when to stop learning
- **select_best()**: selects the best performing model to represent each node of each build; requires a folder hierarchy to be created
- **use_mlp_model()**: predicts values for the data's testing group; returns a list of values for each build
- **use_mlp_model_all()**: predicts values for the data's learning, validation and testing groups; returns a list of values for each build
- **test_model_rmse()**: tests the model's root-mean-square error relative to the entire database; returns a list containing RMSE's for each build
- **test_model_mae()**: tests the model's mean average error relative to the entire database; returns a list containing MAE's for each build
- **test_model_r2()**: tests the model's coefficient of determination, or r-squared value, relative to the entire database; returns a list containing r-squared values for each build
- **output_results(results, which_data, output_filename)**: saves your results to a specified output file; which_data is either 'test' or 'all', for results from 'use_mlp_model()' and 'use_mlp_model_all()' respectively
- **limit_parameters(limited_db_output_filename)**: reduces the input dimensionality of an input database through a "retain the best" process; view the databases directory for the full cetane number database and an example limited database with 15/1666 QSPR descriptors

Working directly with the Server object to handle model creation and data management allows for speedy scripting, but you can still work with the model and data classes directly. View the source code README.md for more information on low-level usage.
