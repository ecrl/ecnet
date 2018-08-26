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

ECNet operates using a **Server** object that interfaces with data utility classes, error calculation functions, and neural network creation classes. The Server object handles importing data and model creation for your project, and serves the data to the model. Configurable variables for neural networks, such as learning rate, number of neurons per hidden layer, activation functions for hidden/input/output layers, and number of training epochs are found in a **.yml** configuration file.

## Configuration .yml file format and variables:

Here is a configuration .yml file we use for cetane number predictions:

```yml
---
learning_rate: 0.1
mlp_hidden_layers:
- - 32
  - relu
- - 32
  - relu
mlp_in_layer_activ: relu
mlp_out_layer_activ: linear
train_epochs: 500
valid_max_epochs: 10000
```

Here are brief explanations of each of these variables:
- **learning_rate**: value passed to the AdamOptimizer to use as its learning rate during training
- **mlp_hidden_layers** - *[[num_neurons_0, layer_type_0],...,[num_neurons_n, layer_type_n]]*: the architecture of the ANN between the input and output layers
	- Rectified linear unit (**'relu'**), **'sigmoid'**, **'softmax'** and **'linear'** *layer_type*s are currently supported
- **mlp_in_layer_activ** - the layer type of the input layer: number of nodes is determined by input data dimensionality
- **mlp_out_layer_activ** - the layer type of the output layer: number of nodes is determined by target data dimensionality
- **train_epochs**: number of training iterations (not used with validation)
- **valid_max_epochs**: the maximum number of training iterations during the validation process

## Server methods:

Here is an overview of the Server object's methods:

- **Server(*config_filename='config.yml', project_file=None*)**: Initialization of Server object - either creates a model configuration file *config_filename*, *or* opens a preexisting *.project* file specified by *project_file*.
- **create_project(*project_name, num_builds=1, num_nodes=5, num_trials=10, print_feedback=True*)**: creates the folder hierarchy for a project with name *project_filename*. Optional variables for the number of builds *num_builds*, number of nodes *num_nodes*, number of trial neural networks per node *num_trials*, and whether to print build status *print_feedback*.
	- note: if this is not called, a project will not be created, and single models will be saved to the 'tmp' folder in your working directory
- **import_data(*data_filename, sort_type='random', data_split=[0.65, 0.25, 0.1]*)**: imports the data from an ECNet formatted CSV database specified in *data_filename*.
	- **sort_type** (either 'random' for random learning, validation and testing set assignments, or 'explicit' for database-specified assignments)
	- **data_split** ([learning %, validation %, testing %] if using random sort_type)
- **limit_parameters(*limit_num, output_filename, use_genetic=False, population_size=500, num_survivors=200, num_generations=25*, shuffle=False)**: reduces the input dimensionality of the currently loaded database to *limit_num* through a "retain the best" algorithm; saves the limited database to *output_filename*. If *use_genetic* is True, a genetic algorithm will be used instead of the retention algorithm; optional arguments for the genetic algorithm are:
	- **population_size** (size of the genetic algorithm population)
	- **num_survivors** (number of population members used to generate the next generation)
	- **num_generations** (number of generations the genetic algorithm runs for)
	- **shuffle** (shuffles learning, validation and test sets for each population member)
- **tune_hyperparameters(*target_score=None, iteration_amt=50, amt_employers=50*)**: optimizes neural network hyperparameters (learning rate, maximum epochs during validation, neuron counts for each hidden layer) using an artifical bee colony algorithm
	- arguments:
		- **target_score** (specify target score for program to terminate)
			- If *None*, ABC will run for *iteration_amt* iterations
		- **iteration_amt** (specify how many iterations to run the colony)
			- Only if *target_score* is not supplied
		- **amt_employers** (specify the amount of employer bees in the colony)
- **train_model(*validate=False, shuffle=None, data_split=[0.65, 0.25, 0.1]*)**: fits neural network(s) to the imported data
	- If validate is **True**, the data's validation set will be used to periodically test model performance to determine when to stop learning up to *valid_max_epochs* config variable epochs; else, trains for *train_epochs* config variable epochs
	- **shuffle** arguments: 
		- **None** (no re-shuffling between trials)
		- **'lv'** (shuffles learning and validation sets between trials)
		- **'lvt'** (shuffles all sets between trials)
	- **data_split** ([learning %, validation %, testing %] if shuffling data)
- **select_best(*dset=None, error_fn='mean_abs_error'*)**: selects the best performing neural network to represent each node of each build; requires a project to be created
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
- **use_model(*dset=None*)**: predicts values for a specified data set; returns a list of results for each build
	- dset arguments: 
		- **None** (defaults to whole dataset)
		- **'learn'** (obtains results for learning set)
		- **'valid'** (obtains results for validation set)
		- **'train'** (obtains results for learning & validation sets)
		- **'test'** (obtains results for test set)
- **calc_error(*args, dset=None*)**: calculates various metrics for error for a specified data set
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
- **save_results(*results, filename*)**: saves your **results** obtained through *use_model()* to a specified output **filename**
- **save_project(*clean_up=True*)**: cleans the project directory (if *clean_up* is True, removing trial neural networks and keeping best neural network from select_best()), copies config and currently loaded dataset into project directory, and creates a '.project' file for later use

Working directly with the Server object to handle model creation and data management allows for speedy scripting, but you can still work with the model and data classes directly. View the source code [README.md](https://github.com/TJKessler/ECNet/tree/master/ecnet) for more information on low-level usage.

## Examples:

To get started, create a Python script to handle your task, and copy an ECNet formatted CSV database file to your working directory. The Server object will create a default configuration file if an existing one is not specified. Example scripts, configuration files, and databases are provided ([examples/config](https://github.com/TJKessler/ECNet/tree/master/examples), [databases](https://github.com/TJKessler/ECNet/tree/master/databases)).

Here is a script for building a project, importing a database, creating and training models for the project, selecting the best model from each build node, grabbing results and errors for the test set, and saving the project:

```python
from ecnet import Server

# Create server object with configuration file 'my_model_configuration.yml' (default config will be generated if the file does not already exist)
sv = Server(config_filename='my_model_configuration.yml')

# Create a project 'my_project', with 10 builds, 5 nodes/build, 75 trials/node
sv.create_project(
    'my_project',
    num_builds=10,
    num_nodes=5,
    num_trials=75,
)

# Import data, randomly assign data set assignments
sv.import_data(
    'my_data.csv',
    sort_type='random',
    data_split=[0.7, 0.2, 0.1]
)

# Trains neural networks using periodic validation, shuffling learn and validate sets between trials
sv.train_model(
    validate=True,
    shuffle='lv',
    data_split=[0.7, 0.2, 0.1]
)

# Select best neural network from each build node (based on test set performance) to predict for the node
sv.select_best(dset='test')

# Predict values for the test data set
test_results = sv.use_mlp_model(dset='test')	

# Output results to specified file
sv.save_results(results=test_results, filename='my_results.csv')	

# Calculates errors for the test set
test_errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error', dset='test')
print(test_errors)

# Save the project to a .project file, removing trial neural networks not selected via select_best()
sv.save_project()

```

You can change all the model configuration variables from your Python script, without having to edit and reopen your configuration .yml file:

```python
from ecnet.server import Server

sv = Server(config_filename='my_model_configuration.yml')

# Configuration variables are found in the server's 'vars' dictionary
sv.vars['learning_rate'] = 0.05
sv.vars['mlp_hidden_layers'] = [[32, 'relu'], [32, 'relu']]
sv.vars['valid_max_epochs'] = 10000
```

Once you save a project, the .project file can be used for predictions:

```python
from ecnet.server import Server

# Specify a 'project_file' argument to open a preexisting project
sv = Server(project_file='my_project.project')

# Open an ECNet database with new data
sv.import_data('new_data.csv')

# Save results to output file
#  - NOTE: no 'dset' argument for 'use_model' defaults to using all currently loaded data
sv.save_results(
    results=sv.use_model(),
    filename='new_data_results.csv'
)
```
To view more examples of common ECNet tasks, view the [examples](https://github.com/TJKessler/ECNet/tree/master/examples) directory.

## Database Format:

ECNet databases are comma-separated value (CSV) formatted files that provide information such as the ID of each molecule, an optional explicit sort type, various strings and groups to identify molecules, and output/target and input parameters. Row 1 is used to identify which columns are used for ID, sorting assignment, various strings and groups, and target and input data.

The [databases](https://github.com/TJKessler/ECNet/tree/master/databases) directory contains databases for cetane number as well as a database template.

## Contributing, Reporting Issues and Other Support:

To contribute to ECNet, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com), Hernan Gelaf-Romer (hernan_gelafromer@student.uml.edu) and/or John Hunter Mack (Hunter_Mack@uml.edu).
