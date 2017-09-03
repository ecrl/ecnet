[![UML Energy & Combustion Research Laboratory](http://faculty.uml.edu/Hunter_Mack/uploads/9/7/1/3/97138798/1481826668_2.png)](http://faculty.uml.edu/Hunter_Mack/)

# ECNet - Large scale machine learning projects for fuel property prediction
### Developed by Travis Kessler with direction from Professor John Hunter Mack
	
**ECNet** is an open source Python package for large scale machine learning projects with a focus on fuel property prediction. A __project__ is considered a collection of __builds__, and each build is a collection of __nodes__. Nodes are averaged to obtain a final predicted value for the build. For each node in a build, multiple neural networks are constructed and the best performing neural network is used for that node's predicted value.

[T. Sennott et al.](https://www.researchgate.net/publication/267576682_Artificial_Neural_Network_for_Predicting_Cetane_Number_of_Biofuel_Candidates_Based_on_Molecular_Structure) have shown that artificial neural networks (ANN's) can be applied to cetane number prediction with relatively little error. ECNet provides scientists an open source tool for predicting key fuel properties of potential next-generation biofuels, reducing the need for costly fuel synthesis and experimentation.

Using ECNet, [T. Kessler et al.](https://www.researchgate.net/publication/317569746_Artificial_neural_network_based_predictions_of_cetane_number_for_furanic_biofuel_additives) have increased the generalizability of ANN's to predict the cetane numbers for a variety of molecular classes represented in the cetane number database, and have increased the accuracy of ANN's for predicting the cetane numbers of underrepresented molecular classes through targeted database expansion.

# Installation:

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

# Usage:

ECNet operates using a **Server** object that interfaces with data utility classes and neural network creation classes. The Server object handles importing data and model creation for your project, and serves the data to the model. Configurable variables like your project's name, number of builds and nodes, ANN learning and architecture variables, data splitting controls, and more are found in a __config.yml__ file in your working directory.

## config.yml format and variables:

Here is what a config.yml file for cetane number prediction looks like:

```yml
---
data_filename: cn_model_v1.0.csv
data_sort_type: random
data_split:
- 0.65
- 0.25
- 0.1
learning_rate: 0.05
mlp_hidden_layers:
- - 32
  - relu
- - 32
  - relu
mlp_in_layer_activ: relu
mlp_out_layer_activ: linear
normals_use: false
project_name: my_project
project_num_builds: 1
project_num_nodes: 1
project_num_trials: 5
project_print_feedback: true
train_epochs: 2500
valid_max_epochs: 7500
valid_mdrmse_memory: 250
valid_mdrmse_stop: 0.00007
```

Here are brief explanations of each of these variables:
- **data_filename**: the location of your formatted .csv database for training and testing data
- **data_sort_type** - random or explicit: how the learning, validation and testing data sets should be chosen
	- Note: explicitly split requires set locations to be defined inside the database
- **data_split** - [learning, validation, testing]: proportions used for the random sort type
- **learning_rate**: value passed to the AdamOptimizer to use as its learning rate during training
- **mlp_hidden_layers** - [[num_neurons_0, layer_type_0],...,[num_neurons_n, layer_type_n]]: the architecture of the ANN between the input and output layers
	- Rectified linear unit ('relu') and Sigmoid ('sigmoid') layer types are currently supported
- **mlp_in_layer_activ** - the layer type of the input layer: number of nodes is determined by data dimensions
- **mlp_out_layer_activ** - the layer type of the output layer: number of nodes is determined by data dimensions
- **normals_use**: boolean to determine if I/O parameters should be normalized (min-max, between 0 and 1)
- **project_name**: the name of your project
- **project_num_builds**: the number of builds in your project
- **project_num_nodes**: the number of nodes in each build
- **project_num_trials**: the number of ANN's to be constructed in each node
- **project_print_feedback**: whether the console will show status messages
- **train_epochs**: number of training iterations (not used with validation)
- **valid_max_epochs**: the maximum number of training iterations during the validation process
- **valid_mdrmse_memory**: how many epochs back the validation process looks in determining the change in validation RMSE over time
- **valid_mdrmse_stop**: the threshold to determine learning cutoff (looks at the change in validation RMSE over time)

## Server methods:

Here is an overview of the Server object's methods:

- **create_save_env()**: creates the folder hierarchy for your project, contained in a folder named after your project name
	- note: if this is not done, a project will not be created, and single models will be saved to the 'tmp' folder in your working directory
- **import_data(*data_filename = None*)**: imports the data from the database specified in 'data_filename', splits the data into learning/validation/testing groups, and packages the data so it's ready to be sent to the model
	- data_filename values: 
		- **None** (default config filename is used)
		- **'database_path_string.csv'** (specified database at location is used)
- **fit_mlp_model(*args*)**: fits multilayer-perceptron(s) to the data, for 'train_epochs' learning iterations
	- arguments: 
		- **None** (no re-shuffling between trials)
		- **'shuffle_lv'** (shuffles learning and validation sets between trials)
		- **'shuffle_lvt'** (shuffles all sets between trials)
- **fit_mlp_model_validation(*args*)**: fits multilayer-perceptron(s) to the data, using the validation set to determine when to stop learning
	- arguments: 
		- **None** (no re-shuffling between trials)
		- **'shuffle_lv'** (shuffles learning and validation sets between trials)
		- **'shuffle_lvt'** (shuffles all sets between trials)
- **select_best()**: selects the best performing model to represent each node of each build; requires a folder hierarchy to be created
- **use_mlp_model(*args*)**: predicts values for the data's testing group; returns a list of values for each build
	- arguments: 
		- **None** (defaults to whole dataset)
		- **'learn'** (obtains results for learning set)
		- **'valid'** (obtains results for validation set)
		- **'train'** (obtains results for learning & validation sets)
		- **'test'** (obtains results for test set)
- **calc_error(*args*, *dset = None*)**: calculates various metrics for error
	- arguments: 
		- **None**, 
		- **'rmse'** (root-mean-squared error)
		- **'r2'** (r-squared value)
		- **'mean_abs_error'** (mean absolute error)
		- **'med_abs_error'** (median absolute error)
	- dset values: 
		- **None** (defaults to calculating error for whole dataset)
		- **'learn'** (errors for learning set)
		- **'valid'** (errors for validation set)
		- **'train'** (errors for learning & validation sets)
		- **'test'** (errors for test set)
- **output_results(*results, output_filename, args*)**: saves your **results** to a specified **output file**
	- arguments:
		- **None** (default to exporting all data)
		- **'learn'** (exports learning set results)
		- **'valid'** (exports validation set results)
		- **'test'** (exports test set results)
- **limit_parameters(*param_num, filename*)**: reduces the input dimensionality of an input database to **param_num** through a "retain the best" process; saves to new database **filename**
- **publish_project()**: cleans the project directory, copies config, normal_params, and currently loaded dataset into the directory, and creates a '.project' file
- **open_project(*project_name*)**: opens a '**project_name**.project' file, importing the project's config, normal_params, and dataset to the Server object

Working directly with the Server object to handle model creation and data management allows for speedy scripting, but you can still work with the model and data classes directly. View the source code README.md for more information on low-level usage.

## Examples:

Here is a script for building a project, importing the dataset, creating models for each build node, training the models, selecting the best model for each build node, grabbing results and errors for the dataset and publishing the project:

```python
from ecnet.server import Server

sv = Server()						# Create server object
sv.create_save_env()					# Create a folder structure for your project
sv.import_data()					# Import data from file specified in config
sv.fit_mlp_model_validation('shuffle_lv')		# Fits model(s), shuffling learn and validate sets between trials
sv.select_best()					# Select best trial from each build node

test_results = sv.use_mlp_model('test')			# Predict values for the test data set
sv.output_results(test_results, 'test_results.csv', 'test')	# Output results to specified file

test_errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error', dset = 'test') # Calculates errors for the test set
print(test_errors)

sv.publish_project()					# Publish the project to a .project file

```

You can change all the configuration variables from your Python script, without having to edit and reopen your config.yml file:

```python
from ecnet.server import Server

sv = Server()
sv.vars['data_filename'] = 'data.csv'
sv.vars['learning_rate'] = 0.05
sv.vars['mlp_hidden_layers'] = [[32, 'relu'], [32, 'relu']]
sv.vars['project_print_feedback'] = False
```

Once you publish a project, the .project file can be opened and used for predictions:

```python
from ecnet.server import Server

sv = Server()
sv.open_project('my_project.project')

sv.import_data('new_data.csv')
results = sv.use_mlp_model()
```
