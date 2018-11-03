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
- Have Python 3.5/3.6 installed
- Have the ability to install Python packages

### Method 1: pip
If you are working in a Linux/Mac environment:
- **sudo pip install ecnet**

Alternatively, in a Windows or virtualenv environment:
- **pip install ecnet**

Note: if multiple Python releases are installed on your system (e.g. 2.7 and 3.6), you may need to execute the correct version of pip. For Python 3.5, change **"pip install ecnet"** to **"pip3 install ecnet"**.

### Method 2: From source
- Download the ECNet repository, navigate to the download location on the command line/terminal, and execute 
**"python setup.py install"**. 

Additional package dependencies (TensorFlow, NumPy PyYaml, ecabc, PyGenetics, ColorLogging) will be installed during the ECNet installation process.

To update your version of ECNet to the latest release version, use "**pip install --upgrade ecnet**".

# Usage:

ECNet operates using a **Server** object that interfaces with data utility classes, error calculation functions, and neural network creation classes. The Server object handles importing data and model creation for your project, and serves the data to the model. Configurable variables for neural networks, such as learning rate, number of neurons per hidden layer, activation functions for hidden/input/output layers, and number of training epochs are found in a **.yml** configuration file.

## Configuration .yml file format and variables

Here is a configuration .yml file we use for cetane number predictions:

```yml
---
learning_rate: 0.1
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
- **hidden_layers** - *[[num_neurons_0, layer_type_0],...,[num_neurons_n, layer_type_n]]*: the architecture of the ANN between the input and output layers
	- Rectified linear unit (**'relu'**), **'sigmoid'**, **'softmax'** and **'linear'** *layer_type*s are currently supported
- **input_activation** - the layer type of the input layer: number of nodes is determined by input data dimensionality
- **output_activation** - the layer type of the output layer: number of nodes is determined by target data dimensionality
- **train_epochs**: number of training iterations (not used with validation)
- **validation_max_epochs**: the maximum number of training iterations during the validation process (if training with periodic validation)

## Using the Server object

To get started, create a Python script to handle your task and copy an ECNet-formatted CSV database file to your working directory. The Server object will create a default configuration file if an existing one is not specified. Example scripts, configuration files, and databases are provided ([examples/config](https://github.com/TJKessler/ECNet/tree/master/examples), [databases](https://github.com/TJKessler/ECNet/tree/master/databases)).

Here is a script for building a project, importing a database, creating and training models for the project, selecting the best model from each build node, grabbing results and errors for the test set, and saving the project. Outlined are a few ways each method can be called:

```python
from ecnet import Server

# Create server object with configuration file 'my_model_configuration.yml' (default config will be
# generated if the file does not already exist)
sv = Server(config_filename='my_model_configuration.yml')

# By default, the Server will log build/selection process to the console and a log file in a local
# "Logs" directory. To turn off logging, specify it in the Server initialization:
sv = Server(config_filename='my_model_configuration.yml', log_progress=False)

# Create a project 'my_project', with 10 builds, 5 nodes/build, 75 trials/node
# If a project is not created, only one neural network will be created when train_model() is called
sv.create_project(
    'my_project',
    num_builds=10,
    num_nodes=5,
    num_trials=75,
)

# Import an ECNet-formatted CSV database, randomly assign data set assignments
sv.import_data(
    'my_data.csv',
    sort_type='random',
    data_split=[0.7, 0.2, 0.1]
)

# To use data set assignments specified in the input database, set the sort_type argument to 'explicit':
sv.import_data(
    'my_data.csv',
    sort_type='explicit'
)

# Trains neural networks using periodic validation, shuffling learn and validate sets between trials
sv.train_model(
    validate=True,
    shuffle='lv',
    data_split=[0.7, 0.2, 0.1]
)

# To avoid shuffling between trials, omit the shuffle variable (defaults to None):
sv.train_model(validate=True)

# If periodic validation is not required, omit the validate variable (defaults to False):
sv.train_model()

# Select best neural network from each build node (based on test set performance) to predict for the node
# Models can be selected based on 'test', 'learn', 'valid' and 'train' (learning and validation) sets,
# or None (default argument, selects based on performance of all sets)
sv.select_best(dset='test')

# Predict values for the test data set
# Results can be obtained from 'test', 'learn', 'valid' and 'train' (learning and validation) sets,
# or None (default argument, obtains results for all sets)
test_results = sv.use_model(dset='test')	

# Output results to specified file
sv.save_results(results=test_results, filename='my_results.csv')	

# Calculates errors for the test set (any combination of these error functions can be supplied as arguments)
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
sv.vars['hidden_layers'] = [[32, 'relu'], [32, 'relu']]
sv.vars['validation_max_epochs'] = 10000
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

# Database Format:

ECNet databases are comma-separated value (CSV) formatted files that provide information such as the ID of each molecule, an optional explicit sort type, various strings and groups to identify molecules, and output/target and input parameters. Row 1 is used to identify which columns are used for ID, sorting assignment, various strings and groups, and target and input data.

The [databases](https://github.com/TJKessler/ECNet/tree/master/databases) directory contains databases for cetane number as well as a database template.

# Contributing, Reporting Issues and Other Support:

To contribute to ECNet, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com), Hernan Gelaf-Romer (hernan_gelafromer@student.uml.edu) and/or John Hunter Mack (Hunter_Mack@uml.edu).
