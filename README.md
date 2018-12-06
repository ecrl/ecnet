[![UML Energy & Combustion Research Laboratory](http://faculty.uml.edu/Hunter_Mack/uploads/9/7/1/3/97138798/1481826668_2.png)](http://faculty.uml.edu/Hunter_Mack/)

# ECNet: scalable, retrainable and deployable machine learning projects for fuel property prediction

[![GitHub version](https://badge.fury.io/gh/tjkessler%2FECNet.svg)](https://badge.fury.io/gh/tjkessler%2FECNet)
[![PyPI version](https://badge.fury.io/py/ecnet.svg)](https://badge.fury.io/py/ecnet)
[![status](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f/status.svg)](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/TJKessler/ECNet/master/LICENSE.txt)
	
**ECNet** is an open source Python package for creating scalable, retrainable and deployable machine learning projects, with a focus on fuel property prediction. An ECNet __project__ is considered a collection of __builds__, and each build is a collection of __nodes__. Nodes are neural networks that have been selected from a pool of candidate neural networks, where the pool's goal is to optimize certain learning criteria (for example, performing optimially on unseen data). Each node contributes a prediction derived from input data, and these predictions are averaged together to calculate the build's final prediction. Using multiple nodes allows a build to learn from a variety of learning and validation sets, which can reduce the build's prediction error. Projects can be saved and reused at a later time allowing additional training and deployable predictive models. 

Future plans for ECNet include:
- distributed candidate training for both CPU and GPU
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

Here is a script for creating a project, importing data, training neural networks, selecting the best neural networks from each build's nodes, grabbing results and errors for the test set, and saving the project. Outlined are a few ways each method can be called:

```python
from ecnet import Server

# Create server object with configuration file 'my_model_configuration.yml' (default config will be
# generated if the file does not already exist)
sv = Server(config_filename='my_model_configuration.yml')

# By default, the Server will log build/selection progress to the console. To disable logging, set
#   the "log" argument to False:
sv = Server(config_filename='my_model_configuration.yml', log=False)

# You can toggle logging with:
sv.log = True
sv.log = False

# If file logging is desired, you can specify a directory to save your logs:
sv = Server(config_filename='my_model_configuration.yml', log_dir='path/to/my/log/directory')

# You can disable file logging with:
sv.log_dir = None

# Or change the directory:
sv.log_dir = 'path/to/my/new/log/directory'

# You can utilize parallel processing (multiprocessing) for model construction (soon), input
#   parameter tuning and hyperparameter optimization:
sv = Server(config_filename='my_model_configuration.yml', num_processes=4)
sv.num_processes = 8

# Create a project 'my_project', with 10 builds, 5 nodes/build, 75 candidates/node
# If a project is not created, only one neural network will be created when train_model() is called
sv.create_project(
    'my_project',
    num_builds=10,
    num_nodes=5,
    num_candidates=75,
)

# Import an ECNet-formatted CSV database, randomly assign data set assignments (proportions of 70%
#   learn, 20% validation, 10% test)
sv.import_data(
    'my_data.csv',
    sort_type='random',
    data_split=[0.7, 0.2, 0.1]
)

# To use data set assignments specified in the input database, set the sort_type argument to
#   'explicit'
sv.import_data(
    'my_data.csv',
    sort_type='explicit'
)

# Trains neural network candidates for all nodes using periodic validation, shuffling learn and
#   validate sets for each candidate
sv.train_model(
    validate=True,
    shuffle='lv',
    data_split=[0.7, 0.2, 0.1]
)

# To avoid shuffling for each candidate, omit the shuffle variable (defaults to None)
sv.train_model(validate=True)

# If periodic validation (determines when to stop training) is not required, omit the validate
#   variable (defaults to False)
sv.train_model()

# Select best neural network from each build's nodes (based on test set performance) to predict
#   for the node Models can be selected based on 'test', 'learn', 'valid' and 'train' (learning
#   and validation) sets, or None (selects based on performance of all sets)
sv.select_best(dset='test')

# Predict values for the test data set
# Results can be obtained from 'test', 'learn', 'valid' and 'train' (learning and validation) sets,
# or None (obtains results for all sets)
test_results = sv.use_model(dset='test')	

# Output results to specified file
sv.save_results(results=test_results, filename='my_results.csv')	

# Calculates errors for the test set (any combination of these error functions can be supplied as
#   arguments)
test_errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error', dset='test')
print(test_errors)

# Save the project to a .project file, removing candidate neural networks not selected via
#   select_best()
sv.save_project()

# To retain candidate neural networks, set the clean_up argument to False
sv.save_project(clean_up=False)

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

Once you save a project, the .project file can be used at a later time:

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
To view more examples of common ECNet tasks such as hyperparameter optimization and input dimensionality reduction, view the [examples](https://github.com/TJKessler/ECNet/tree/master/examples) directory. For additional documentation on Server methods view the README in the [ecnet](https://github.com/tjkessler/ecnet/tree/master/ecnet) directory.

# Database Format:

ECNet databases are comma-separated value (CSV) formatted files that provide information such as the ID of each data point, an optional explicit sort type, various strings and groups to identify data points, target values and input parameters. Row 1 is used to identify which columns are used for ID, explicit sorting assignment, various strings and groups, and target and input data, and row 2 contains the names of these strings/groups/targets/inputs. Additional rows are data points.

The [databases](https://github.com/TJKessler/ECNet/tree/master/databases) directory contains databases for cetane number as well as a database template.

# Contributing, Reporting Issues and Other Support:

To contribute to ECNet, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com), Hernan Gelaf-Romer (hernan_gelafromer@student.uml.edu) and/or John Hunter Mack (Hunter_Mack@uml.edu).
