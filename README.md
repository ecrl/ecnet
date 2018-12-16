[![UML Energy & Combustion Research Laboratory](http://faculty.uml.edu/Hunter_Mack/uploads/9/7/1/3/97138798/1481826668_2.png)](http://faculty.uml.edu/Hunter_Mack/)

# ECNet: scalable, retrainable and deployable machine learning projects for fuel property prediction

[![GitHub version](https://badge.fury.io/gh/tjkessler%2FECNet.svg)](https://badge.fury.io/gh/tjkessler%2FECNet)
[![PyPI version](https://badge.fury.io/py/ecnet.svg)](https://badge.fury.io/py/ecnet)
[![status](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f/status.svg)](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/TJKessler/ECNet/master/LICENSE.txt)
	
**ECNet** is an open source Python package for creating scalable, retrainable and deployable machine learning projects, with a focus on fuel property prediction. An ECNet __project__ is considered a collection of __builds__, and each build is a collection of __nodes__. Nodes are neural networks that have been selected from a pool of candidate neural networks, where the pool's goal is to optimize certain learning criteria (for example, performing optimially on unseen data). Each node contributes a prediction derived from input data, and these predictions are averaged together to calculate the build's final prediction. Using multiple nodes allows a build to learn from a variety of learning and validation sets, which can reduce the build's prediction error. Projects can be saved and reused at a later time allowing additional training and deployable predictive models. 

Future plans for ECNet include:
- distributed candidate training for GPU's
- a graphical user interface
- implementing neural network neuron diagnostics - maybe it's not a black box after all 🤔

[T. Sennott et al.](https://doi.org/10.1115/ICEF2013-19185) have shown that neural networks can be applied to cetane number prediction with relatively little error. ECNet provides scientists an open source tool for predicting key fuel properties of potential next-generation biofuels, reducing the need for costly fuel synthesis and experimentation.

Using ECNet, [T. Kessler et al.](https://doi.org/10.1016/j.fuel.2017.06.015) have increased the generalizability of neural networks to predict the cetane number for a variety of molecular classes represented in our [cetane number database](https://github.com/TJKessler/ECNet/tree/master/databases), and have increased the accuracy of neural networks for predicting the cetane number of underrepresented molecular classes through targeted database expansion.

# Installation:

### Prerequisites:
- Have Python 3.5/3.6 installed
- Have the ability to install Python packages

### Method 1: pip
If you are working in a Linux/Mac environment:
- **sudo pip install ecnet**

Alternatively, in a Windows or virtualenv environment:
- **pip install ecnet**

Note: if multiple Python releases are installed on your system (e.g. 2.7 and 3.6), you may need to execute the correct version of pip. For Python 3.6, change **"pip install ecnet"** to **"pip3 install ecnet"**.

### Method 2: From source
- Download the ECNet repository, navigate to the download location on the command line/terminal, and execute 
**"python setup.py install"**. 

Additional package dependencies (TensorFlow, NumPy, PyYaml, ecabc, PyGenetics, ColorLogging) will be installed during the ECNet installation process. If raw performance is your thing, consider building numerical packages like TensorFlow and NumPy from source.

To update your version of ECNet to the latest release version, use "**pip install --upgrade ecnet**".

# Usage:

ECNet operates using a **Server** object that interfaces with data utility classes, error calculation functions, and neural network creation classes. The Server object handles importing data and model creation for your project, and serves the data to the model. Configurable variables for neural networks, such as learning rate, number of neurons per hidden layer, activation functions for hidden/input/output layers, and number of training epochs are found in a **.yml** configuration file.

## Configuration .yml file format and variables

Here is a configuration .yml file we use for cetane number predictions:

```yml
---
learning_rate: 0.1
keep_prob: 1.0
hidden_layers:
- - 32
  - relu
- - 32
  - relu
input_activation: relu
output_activation: linear
train_epochs: 500
validation_max_epochs: 10000
```

Here are brief explanations of each of these variables:
- **learning_rate**: value passed to the AdamOptimizer to use as its learning rate during training
- **keep_prob**: probability that a neuron in the hidden layers is not subjected to dropout
- **hidden_layers**: *[[num_neurons_0, layer_type_0],...,[num_neurons_n, layer_type_n]]*: the architecture of the neural network between input and output layers
	- Rectified linear unit (**'relu'**), **'sigmoid'**, **'softmax'** and **'linear'** *layer_type*s are currently supported
- **input_activation**: the layer type of the input layer: number of nodes is determined by input data dimensionality
- **output_activation**: the layer type of the output layer: number of nodes is determined by target data dimensionality
- **train_epochs**: number of training iterations (not used with validation)
- **validation_max_epochs**: the maximum number of training iterations during the validation process (if training with periodic validation)

## Using the Server object

To get started, create a Python script to handle your task and copy an ECNet-formatted CSV database file to your working directory. The Server object will create a default configuration file if an existing one is not specified or found. Example scripts, configuration files, and databases are provided ([examples/config](https://github.com/TJKessler/ECNet/tree/master/examples), [databases](https://github.com/TJKessler/ECNet/tree/master/databases)).

Your first steps are importing the Server object, initializing the Server and importing some data:

```python
from ecnet import Server

# Initialize a Server
sv = Server()

# If 'config.yml' does not already exist in your working directory, it will be created with
#   default values; to specify another configuration file, use the config_filename argument
sv = Server(config_filename='my_model_configuration.yml')

# By default, the Server will log build/selection progress to the console. To disable logging,
#   set the "log_level" argument to 'disable':
sv = Server(config_filename='my_model_configuration.yml', log_level='disable')

# You can set the log level to 'disable', 'debug', 'info', 'warn', 'error', 'crit'; the level
#   is set to info by default
sv.log_level = 'debug'

# If file logging is desired, you can specify a directory to save your logs:
sv = Server(config_filename='my_model_configuration.yml', log_dir='path/to/my/log/directory')

# File logging is disabled until a log_dir is supplied; you can disable file logging with:
sv.log_dir = None

# Or change the directory:
sv.log_dir = 'path/to/my/new/log/directory'

# You can utilize parallel processing (multiprocessing) for model construction, input parameter
#   tuning and hyperparameter optimization:
sv = Server(config_filename='my_model_configuration.yml', num_processes=4)
sv.num_processes = 8

# Import an ECNet-formatted CSV database, randomly assign data set assignments (proportions of
#   70% learn, 20% validation, 10% test)
sv.import_data(
    'my_data.csv',
    sort_type='random',
    data_split=[0.7, 0.2, 0.1]
)

# You can specify set assignments in an ECNet formatted database; to use data set assignments
#   specified in the input database, set the sort_type argument to 'explicit'
sv.import_data(
    'my_data.csv',
    sort_type='explicit'
)
```

You can change all the model configuration variables from your Python script, without having to edit and reopen your configuration .yml file:

```python
from ecnet.server import Server

sv = Server(config_filename='my_model_configuration.yml')

# Configuration variables are found in the server's 'vars' dictionary
sv.vars['learning_rate'] = 0.05
sv.vars['keep_prob'] = 0.75
sv.vars['hidden_layers'] = [[32, 'relu'], [32, 'relu']]
sv.vars['validation_max_epochs'] = 10000
```

Optimal input dimensionality, i.e. finding a balance between runtime and precision/accuracy, is often beneficial. ECNet has a few tools to help out with this. To limit input dimensionality to a specified number of input parameters, ECNet utilizes an iterative inclusion (add, pair and retain) method and a genetic algorithm:

```python
# To limit input dimensionality using iterative inclusion, call limit_input_parameters and
#   supply a desired dimension
sv.limit_input_parameters(15)

# You can save an ECNet-formatted database once limiting is complete
sv.limit_input_parameters(15, output_filename='my_limited_database.csv')

# To limit input dimensionality using a genetic algorithm, supply the use_genetic argument,
#   the size of the population and the number of generations to run the algorithm for
sv.limit_input_parameters(
    15,
    use_genetic=True,
    population_size=50,
    num_generations=10
)

# You have the option to shuffle learning/validation/testing data for each population member
sv.limit_input_parameters(
    15,
    use_genetic=True,
    population_size=50,
    num_generations=10,
    shuffle=True,
    data_split=[0.7, 0.2, 0.1]
)
```

Optimal hyperparameters are essential for mapping inputs to outputs during neural network
training. ECNet utilizes an artificial bee colony, [ECabc](https://github.com/ecrl/ecabc), to optimize hyperparameters such as
learning rate, dropout rate, maximum number of epochs during validation training, and the size
(number of neurons) of each hidden layer.


```python
# Tune hyperparameters for 50 iterations (bee cycles) with 50 employer bees
sv.tune_hyperparameters(num_iterations=50, num_employers=50)

# The fitness function handed to the bees calculates mean absolute error; you can halt tuning
#   if the mean absolute error falls below a specified threshold
sv.tune_hyperparameters(target_score='2.5', num_employers=50)
```

ECNet is able to create an ensemble of neural networks (candidates chosen from nodes) to
predict for a final model (build). A runtime with multiple builds/nodes/candidates is called
a project. Projects can be saved and used at a later time.


```python
# Create a project 'my_project' with 10 builds, 5 nodes/build, and 75 candidates/node; if this
#   method is not called, only one neural network will be created during training
sv.create_project(
    'my_project',
    num_builds=10,
    num_nodes=5,
    num_candidates=75,
)

# Train neural networks using the number of epochs in your configuration file
sv.train_model()

# To use periodic validation (training halts when validation set performance stops improving),
#   supply the validate argument
sv.train_model(validate=True)

# We can shuffle either 'train' (learning and validation) or 'all' sets with the shuffle
#   argument and a data split
sv.train_model(
    validation=True,
    shuffle='train',
    data_split=[0.7, 0.2, 0.1]
)

# If you are using a project, select the best performing neural networks based on test set
#   performance
sv.select_best(dset='test')

# You can select based on 'learn', 'valid', 'train', 'test' or None (all) set performances
sv.select_best(dset='train')

# Save your project
sv.save_project()

# You can save it with a name other than the one assigned
sv.save_project(filename='path/to/my/save.prj')

# A project will have a folder structure with all your neural networks in your working directory;
#   if your script is complete and you are done with the project, you can remove it (don't use
#   this if you want to use the models in the same script!)
sv.save_project(clean_up=True)

# Predict values for the test data set
test_results = sv.use_model(dset='test')

# You can predict for 'learn', 'valid', 'train', 'test', or None (all) sets
test_results = sv.use_model(dset='train')

# If you want to save these results to a CSV file, supply the output_filename argument
sv.use_model(dset='test', output_filename='my/test/results.csv')

# Calculates errors for the test set (any combination of these error functions can be supplied as
#   arguments, and any dset listed above)
test_errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error', dset='test')
print(test_errors)
```

Once you save a project, the .prj file can be used at a later time:

```python
from ecnet.server import Server

# Specify a 'project_file' argument to open a preexisting project
sv = Server(project_file='my_project.prj')

# Open an ECNet database with new data
sv.import_data('new_data.csv')

# Save results to output file
#  - NOTE: no 'dset' argument for 'use_model' defaults to using all currently loaded data
sv.use_model(output_filename='my/new/test/results.csv')
```
To view more examples of common ECNet tasks such as hyperparameter optimization and input dimensionality reduction, view the [examples](https://github.com/tjKessler/ecnet/tree/master/examples) directory. For additional documentation on Server methods and lower-level usage view the README in the [ecnet](https://github.com/tjkessler/ecnet/tree/master/ecnet) directory.

# Database Format:

ECNet databases are comma-separated value (CSV) formatted files that provide information such as the ID of each data point, an optional explicit sort type, various strings and groups to identify data points, target values and input parameters. Row 1 is used to identify which columns are used for ID, explicit sorting assignment, various strings and groups, and target and input data, and row 2 contains the names of these strings/groups/targets/inputs. Additional rows are data points.

The [databases](https://github.com/TJKessler/ECNet/tree/master/databases) directory contains databases for cetane number as well as a database template.

# Contributing, Reporting Issues and Other Support:

To contribute to ECNet, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com), Hernan Gelaf-Romer (hernan_gelafromer@student.uml.edu) and/or John Hunter Mack (Hunter_Mack@uml.edu).
