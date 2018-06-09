[![UML Energy & Combustion Research Laboratory](http://faculty.uml.edu/Hunter_Mack/uploads/9/7/1/3/97138798/1481826668_2.png)](http://faculty.uml.edu/Hunter_Mack/)

# ECNet: Large scale machine learning projects for fuel property prediction

[![GitHub version](https://badge.fury.io/gh/tjkessler%2FECNet.svg)](https://badge.fury.io/gh/tjkessler%2FECNet)
[![PyPI version](https://badge.fury.io/py/ecnet.svg)](https://badge.fury.io/py/ecnet)
[![status](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f/status.svg)](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/TJKessler/ECNet/master/LICENSE.txt)
	
**ECNet** is an open source Python package for creating large scale machine learning projects with a focus on fuel property prediction. A __project__ is considered a collection of __builds__, and each build is a collection of __nodes__. Nodes are averaged to obtain a final predicted value for the build. For each node in a build, multiple neural networks are constructed and the best performing neural network is used as that node's predictor. Using multiple nodes allows a build to learn from multiple learning and validation sets, reducing the build's error. Projects can be saved and reused.

[T. Sennott et al.](https://doi.org/10.1115/ICEF2013-19185) have shown that artificial neural networks (ANN's) can be applied to cetane number prediction with relatively little error. ECNet provides scientists an open source tool for predicting key fuel properties of potential next-generation biofuels, reducing the need for costly fuel synthesis and experimentation.

Using ECNet, [T. Kessler et al.](https://doi.org/10.1016/j.fuel.2017.06.015) have increased the generalizability of ANN's to predict the cetane numbers for molecules from a variety of molecular classes represented in the [cetane number database](https://github.com/TJKessler/ECNet/tree/master/databases), and have increased the accuracy of ANN's for predicting the cetane numbers for molecules from underrepresented molecular classes through targeted database expansion.

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

Additional package dependencies (TensorFlow, PyYaml, ecabc, PyGenetics) will be installed during the ECNet installation process.

To update your version of ECNet to the latest release version, use "**pip install --upgrade ecnet**".

# Usage:

ECNet operates using a **Server** object that interfaces with data utility classes, error calculation functions, and neural network creation classes. The Server object handles importing data and model creation for your project, and serves the data to the model. Configurable variables like your project's name, number of builds and nodes, ANN learning and architecture variables, data splitting controls, and more are found in a __config.yml__ file in your working directory.

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
learning_rate: 0.1
mlp_hidden_layers:
- - 32
  - relu
- - 32
  - relu
mlp_in_layer_activ: relu
mlp_out_layer_activ: linear
project_name: my_project
project_num_builds: 10
project_num_nodes: 5
project_num_trials: 75
project_print_feedback: true
train_epochs: 500
valid_max_epochs: 15000
```

Here are brief explanations of each of these variables:
- **data_filename**: the location of your formatted .csv database for training and testing data
- **data_sort_type** - *random or explicit*: how the learning, validation and testing data sets should be chosen
	- Note: explicitly split requires set locations to be defined inside the database
- **data_split** - *[learning, validation, testing]*: proportions used for the random sort type
- **learning_rate**: value passed to the AdamOptimizer to use as its learning rate during training
- **mlp_hidden_layers** - *[[num_neurons_0, layer_type_0],...,[num_neurons_n, layer_type_n]]*: the architecture of the ANN between the input and output layers
	- Rectified linear unit (**'relu'**), **'sigmoid'**, and **'linear'** *layer_type*s are currently supported
- **mlp_in_layer_activ** - the layer type of the input layer: number of nodes is determined by data dimensions
- **mlp_out_layer_activ** - the layer type of the output layer: number of nodes is determined by data dimensions
- **project_name**: the name of your project
- **project_num_builds**: the number of builds in your project
- **project_num_nodes**: the number of nodes in each build
- **project_num_trials**: the number of ANN's to be constructed in each node
- **project_print_feedback**: whether the console will show status messages
- **train_epochs**: number of training iterations (not used with validation)
- **valid_max_epochs**: the maximum number of training iterations during the validation process

## Server methods:

Here is an overview of the Server object's methods:

- **create_project(*project_name = None*)**: creates the folder hierarchy for your project
	- project_name values:
		- **None** (default config project name is used)
		- **Other** (supplied project name is used for your project)
	- note: if this is not called, a project will not be created, and single models will be saved to the 'tmp' folder in your working directory
- **import_data(*data_filename = None*)**: imports the data from the database specified in *data_filename*, splits the data into learning/validation/testing groups, and packages the data so it's ready to be sent to the model
	- data_filename values: 
		- **None** (default config filename is used)
		- **Other** (supplied CSV database will be used)
- **limit_parameters(*limit_num, output_filename, use_genetic = False, population_size = 100, num_survivors = 33, num_generations = 10*)**: reduces the input dimensionality of an input database to *limit_num* through a "retain the best" algorithm; saves the limited database to *output_filename*. If *use_genetic* is True, a genetic algorithm will be used instead of the retention algorithm; optional arguments for the genetic algorithm are:
	- **population_size** (size of the genetic algorithm population)
	- **num_survivors** (number of population members used to generate the next generation)
	- **num_generations** (number of generations the genetic algorithm runs for)
- **tune_hyperparameters(*target_score = None, iteration_amt = 50, amt_employers = 50*)**: optimizes neural network hyperparameters (learning rate, maximum epochs during validation, neuron counts for each hidden layer) using an artifical bee colony algorithm
	- arguments:
		- **target_score** (specify target score for program to terminate)
			- If *None*, ABC will run for *iteration_amt* iterations
		- **iteration_amt** (specify how many iterations to run the colony)
			- Only if *target_score* is not supplied
		- **amt_employers** (specify the amount of employer bees in the colony)
- **train_model(*args, validate = False*)**: fits neural network(s) to the imported data
	- arguments: 
		- **None** (no re-shuffling between trials)
		- **'shuffle_lv'** (shuffles learning and validation sets between trials)
		- **'shuffle_lvt'** (shuffles all sets between trials)
	- If validate is **True**, the data's validation set will be used to periodically test model performance to determine when to stop learning; else, trains for *train_epochs* iterations
- **select_best(*dset = None, error_fn = 'rmse'*)**: selects the best performing neural network to represent each node of each build; requires a folder hierarchy to be created
	- dset arguments:
		- **None** (best performers are based on entire database)
		- **'learn'** (best performers are based on learning set)
		- **'valid'** (best performers are based on validation set)
		- **'train'** (best performers are based on learning & validation sets)
		- **'test'** (best performers are based on test set)
	- error_fn arguments:
		- **'rmse'** (RMSE is used as the metric to determine best performing neural network)
		- **'mean_abs_error'** (Mean absolute error is used as the metric to determine best performing neural network)
		- **'med_abs_error'** (Median absolute error is used as the metric to determine best performing neural network)
- **use_model(*dset = None*)**: predicts values for a specified data set; returns a list of results for each build
	- dset arguments: 
		- **None** (defaults to whole dataset)
		- **'learn'** (obtains results for learning set)
		- **'valid'** (obtains results for validation set)
		- **'train'** (obtains results for learning & validation sets)
		- **'test'** (obtains results for test set)
- **calc_error(*args*, *dset = None*)**: calculates various metrics for error for a specified data set
	- arguments: 
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
- **output_results(*results, filename = 'my_results.csv'*)**: saves your **results** to a specified output **filename**
- **save_project(*clean_up = True*)**: cleans the project directory (if *clean_up* is True) (removes neural networks not selected via *select_best()*, copies config and currently loaded dataset into the directory, and creates a '.project' file
- **open_project(*filename*)**: opens a '*filename*.project' file, importing the project's config, dataset, and trained models to the Server object

Working directly with the Server object to handle model creation and data management allows for speedy scripting, but you can still work with the model and data classes directly. View the source code [README.md](https://github.com/TJKessler/ECNet/tree/master/ecnet) for more information on low-level usage.

## Examples:

To get started, create a Python script and a config.yml file to handle your task, and copy a formatted database (.csv) file to your working directory. The Server object will create a default configuration file if none are provided. Example scripts, configuration files, and databases are provided ([examples/config](https://github.com/TJKessler/ECNet/tree/master/examples), [databases](https://github.com/TJKessler/ECNet/tree/master/databases)).

Here is a script for building a project, importing the dataset, creating and training models for each build node, selecting the best model from each build node, grabbing results and errors for the test set, and publishing the project:

```python
from ecnet.server import Server

# Create server object
sv = Server()

# Create a project
sv.create_project('my_new_project')

# Import data
sv.import_data('my_data.csv')

# Trains neural networks using periodic validation, shuffling learn and validate sets between trials
sv.train_model('shuffle_lv', validate = True)

# Select best neural network from each build node (based on test set performance) to predict for the node
sv.select_best('test')

# Predict values for the test data set
test_results = sv.use_mlp_model('test')	

# Output results to specified file
sv.output_results(results = test_results, filename = 'test_results.csv')	

# Calculates errors for the test set
test_errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error', dset = 'test')
print(test_errors)

# Save the project to a .project file
sv.save_project()

```

You can change all the configuration variables from your Python script, without having to edit and reopen your config.yml file:

```python
from ecnet.server import Server

sv = Server()

# Configuration variables are found in the server's 'vars' dictionary
sv.vars['data_filename'] = 'data.csv'
sv.vars['learning_rate'] = 0.05
sv.vars['mlp_hidden_layers'] = [[32, 'relu'], [32, 'relu']]
sv.vars['project_print_feedback'] = False
```

Once you save a project, the .project file can be opened and used for predictions:

```python
from ecnet.server import Server

sv = Server()

# Opens a pre-existing project
sv.open_project('my_project.project')

# Open a new dataset
sv.import_data('new_data.csv')

# Save results to output file
#  - NOTE: no 'dset' argument for 'use_model' defaults to using all currently loaded data
sv.output_results(results = sv.use_model(), filename = 'new_data_results.csv')
```
To view more examples of common ECNet tasks, view the [examples](https://github.com/TJKessler/ECNet/tree/master/examples) directory.

## Database Format:

ECNet databases are comma-separated value (CSV) formatted files that provide information such as the ID of each molecule, an optional explicit sort type, various strings and groups to identify molecules, and output/target and input parameters. Row 1 is used to identify which columns are used for ID, sorting assignment, various strings and groups, and target and input data.

The [databases](https://github.com/TJKessler/ECNet/tree/master/databases) directory contains databases for cetane number as well as a database template.

## Contributing, Reporting Issues and Other Support:

To contribute to ECNet, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com), Hernan Gelaf-Romer (hernan_gelafromer@student.uml.edu) and/or John Hunter Mack (Hunter_Mack@uml.edu).
