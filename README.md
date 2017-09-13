[![UML Energy & Combustion Research Laboratory](http://faculty.uml.edu/Hunter_Mack/uploads/9/7/1/3/97138798/1481826668_2.png)](http://faculty.uml.edu/Hunter_Mack/)

# ECNet: Large scale machine learning projects for fuel property prediction
	
**ECNet** is an open source Python package for creating large scale machine learning projects with a focus on fuel property prediction. A __project__ is considered a collection of __builds__, and each build is a collection of __nodes__. Nodes are averaged to obtain a final predicted value for the build. For each node in a build, multiple neural networks are constructed and the best performing neural network is used as that node's predictor.

Here is a visual represntation of a build for cetane number prediction:
![Build Diagram](https://github.com/TJKessler/ECNet/blob/master/misc/build_figure.png)

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

Additional package dependencies (TensorFlow, PyYaml) will be installed during the ECNet installation process.

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
- **data_sort_type** - *random or explicit*: how the learning, validation and testing data sets should be chosen
	- Note: explicitly split requires set locations to be defined inside the database
- **data_split** - *[learning, validation, testing]*: proportions used for the random sort type
- **learning_rate**: value passed to the AdamOptimizer to use as its learning rate during training
- **mlp_hidden_layers** - *[[num_neurons_0, layer_type_0],...,[num_neurons_n, layer_type_n]]*: the architecture of the ANN between the input and output layers
	- Rectified linear unit (**'relu'**), **'sigmoid'**, and **'linear'** *layer_type*s are currently supported
- **mlp_in_layer_activ** - the layer type of the input layer: number of nodes is determined by data dimensions
- **mlp_out_layer_activ** - the layer type of the output layer: number of nodes is determined by data dimensions
- **normals_use**: *boolean* to determine if I/O parameters should be normalized (min-max, between 0 and 1)
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
- **use_mlp_model(*args*)**: predicts values for the data's testing group; returns a list of results for each build
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
- **output_results(*results, output_filename*)**: saves your **results** to a specified **output file**
- **limit_parameters(*param_num, filename*)**: reduces the input dimensionality of an input database to **param_num** through a "retain the best" process; saves to new database **filename**
- **publish_project()**: cleans the project directory, copies config, normal_params, and currently loaded dataset into the directory, and creates a '.project' file
- **open_project(*project_name*)**: opens a '**project_name**.project' file, importing the project's config, normal_params, and dataset to the Server object

Working directly with the Server object to handle model creation and data management allows for speedy scripting, but you can still work with the model and data classes directly. View the source code README.md for more information on low-level usage.

## Examples:

To get started, create a Python script and a config.yml file to handle your task, and copy a formatted database (.csv) file to your working directory. The Server object will create a default configuration file if none are provided. Example scripts, configuration files, and databases are provided ([examples/config](https://github.com/TJKessler/ECNet/tree/master/examples), [databases](https://github.com/TJKessler/ECNet/tree/master/databases)).

Here is a script for building a project, importing the dataset, creating and training models for each build node, selecting the best model from each build node, grabbing results and errors for the test set, and publishing the project:

```python
from ecnet.server import Server

# Create server object
sv = Server()

# Create a folder structure for your project
sv.create_save_env()

# Import data from file specified in config
sv.import_data()

# Fits model(s), shuffling learn and validate sets between trials
sv.fit_mlp_model_validation('shuffle_lv')

# Select best trial from each build node to predict for the node
sv.select_best()

# Predict values for the test data set
test_results = sv.use_mlp_model('test')	

# Output results to specified file
sv.output_results(test_results, 'test_results.csv')	

# Calculates errors for the test set
test_errors = sv.calc_error('rmse','r2','mean_abs_error','med_abs_error', dset = 'test')
print(test_errors)

# Publish the project to a .project file
sv.publish_project()

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

Once you publish a project, the .project file can be opened and used for predictions:

```python
from ecnet.server import Server

sv = Server()

# Opens a pre-existing project
sv.open_project('my_project.project')

# Open a new dataset
sv.import_data('new_data.csv')

# Save results to output file
sv.output_results(results = sv.use_mlp_model(), 'new_data_results.csv')
```
To view more examples of common ECNet tasks, view the [examples](https://github.com/TJKessler/ECNet/tree/master/examples) directory.

## Database Format:

ECNet databases are comma-separated value (CSV) formatted files that provide information such as the ID of each molecule (DATAid), an optional explicit sort type (T/V/L), various strings and groups to identify molecules, and input and output/target parameters. The number of strings, groups, outputs/targets and specific DATAID's to automatically drop are determined by the master parameters in rows 1-2. Row 3 contains the headers for each sub-section (DATAID, T/V/L, strings, groups, paramters), and row 4 contains specific string, group, and parameter names. The number of outputs/targets, determined by the "Num of Outputs" master parameter, tells the data importing software how many parameter columns (from left to right) are used as outputs/targets.

The [databases](https://github.com/TJKessler/ECNet/tree/master/databases) directory contains databases for cetane number as well as a database template.

## Contributing, Reporting Issues and Other Support:

To contribute to ECNet, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (Travis_Kessler@student.uml.edu) or John Hunter Mack (Hunter_Mack@uml.edu).
