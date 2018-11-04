# Server methods, and low-level usage of model, data_utils, error_utils, and limit_parameters

## server.py
#### Class: Server

Methods:
- **Server(*config_filename='config.yml', project_file=None, log_progress=True*)**: Initialization of Server object - either creates a model configuration file *config_filename*, *or* opens a preexisting *.project* file specified by *project_file*. Will log training, selection, limiting, etc. progress to the console and a log file if *log_progress* == True
- **create_project(*project_name, num_builds=1, num_nodes=5, num_candidates=10*)**: creates the folder hierarchy for a project with name *project_filename*. Optional variables for the number of builds *num_builds*, number of nodes *num_nodes*, number of candidate neural networks per node *num_candidates*
	- note: if this is not called, a project will not be created, and single models will be saved to the 'tmp' folder in your working directory
- **import_data(*data_filename, sort_type='random', data_split=[0.65, 0.25, 0.1]*)**: imports the data from an ECNet formatted CSV database specified in *data_filename*.
	- **sort_type** (either 'random' for random learning, validation and testing set assignments, or 'explicit' for database-specified assignments)
	- **data_split** ([learning %, validation %, testing %] if using random sort_type)
- **limit_input_parameters(*limit_num, output_filename, use_genetic=False, population_size=500, num_generations=25, num_processes=0, shuffle=False, data_split=[0.65, 0.25, 0.1]*)**: reduces the input dimensionality of the currently loaded database to *limit_num* through a "retain the best" algorithm; saves the limited database to *output_filename*. If *use_genetic* is True, a genetic algorithm will be used instead of the retention algorithm; optional arguments for the genetic algorithm are:
	- **population_size** (size of the genetic algorithm population)
	- **num_generations** (number of generations the genetic algorithm runs for)
	- **num_processes** (if your machine supports it, number of parallel processes the genetic algorithm will utilize)
	- **shuffle** (shuffles learning, validation and test sets for each population member)
    - **data_split** (if shuffle == True, these are data set proportions)
- **tune_hyperparameters(*target_score=None, num_iterations=50, num_employers=50*)**: optimizes neural network hyperparameters (learning rate, maximum epochs during validation, neuron counts for each hidden layer) using an artifical bee colony algorithm
	- arguments:
		- **target_score** (specify target score for program to terminate)
			- If *None*, ABC will run for *num_iterations* iterations
		- **num_iterations** (specify how many iterations to run the colony)
			- Only if *target_score* is not supplied
		- **num_employers** (specify the amount of employer bees in the colony)
- **train_model(*validate=False, shuffle=None, data_split=[0.65, 0.25, 0.1]*)**: fits neural network(s) to the imported data
	- If validate is **True**, the data's validation set will be used to periodically test model performance to determine when to stop learning up to *validation_max_epochs* config variable epochs; else, trains for *train_epochs* config variable epochs
	- **shuffle** arguments: 
		- **None** (no re-shuffling data for each neural network)
		- **'lv'** (shuffles learning and validation sets data for each neural network)
		- **'lvt'** (shuffles all sets data for each neural network)
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
- **save_project(*clean_up=True*)**: cleans the project directory (if *clean_up* is True, removing candidate neural networks and keeping best neural network from select_best()), copies config and currently loaded dataset into project directory, and creates a '.project' file for later use

## model.py
#### Class: MultilayerPerceptron

Methods:
- **add_layer(size, act_fn)**: appends a *Layer* to the MLP's layer list
  - supported activation functions: 'relu', 'sigmoid', 'linear', 'softmax'
- **connect_layers()**: initializes TensorFlow variables for weights and biases between each layer; fully connected
- **fit(x_l, y_l, learning_rate, train_epochs)**: fits the MLP to the inputs (**x_l**) and outputs (**y_l**) for **train_epochs** iterations with a learning rate of **learning_rate**
- **fit_validation(x_l, x_v, y_l, y_v, learning_rate, max_epochs)**: fits the MLP, periodically checking MLP performance using validation data; learning is stopped when validation data performance stops improving
  - **max_epochs** is the cutoff point if validation cutoff has not been reached
- **use(x)**: used to pass data through the trained model to get a prediction; returns predicted values
- **save(filepath)**: saves the TensorFlow session (.sess) and model architecture information (.struct) to specified filename
- **load(filepath)**: opens a TensorFlow session (.sess) and model architecture information (.struct) from specified filename

## data_utils.py
#### Class: DataFrame
Methods:
- **DataFrame(filename)**: imports a formatted database, creates DataPoints for each data entry, grabs string and group names and counts
- **create_sets(random=False, split=None)**: create learning, validation and testing sets from database set assignments; if *random* == True, random set assignments are assigned with a split of *split=[learn%, valid%, test%]*
- **create_sorted_sets(sort_string, split)**: using *sort_string*, a string contained in the DataFrame's imported database, assigns proportions *split=[learn%, valid%, test%]* of each possible string value to learning, validation and testing sets
- **shuffle(args*, split)**: shuffles data for specified sets with *split=[learn%, valid%, test%]* set assignments
   - args combinations:
    - 'l', 'v', 't' (shuffles data for learning, validation and testing sets)
    - 'l', 'v' (shuffles data for learning and validation sets)
- **package_sets()** returns a PackagedData object, containing NumPy arrays for learning, validation and testing input and target sets

Functions:
- **save_results(results, DataFrame, filename)**: outputs *results* (calculated by model.py for a specified data set) to *filename*; *DataFrame* is required for outputting data entry names, strings, groups, etc.

## error_utils.py
Notation:
- **y_hat**: predicted values
- **y**: known/training values
  
Error Functions:
- **calc_rmse(y_hat, y)**: returns the root-mean-squared error
- **calc_mean_abs_error(y_hat, y)**: returns the mean absolute error
- **calc_med_abs_error(y_hat, y)**: returns the median absolute error
- **calc_r2(y_hat, y)**: returns the correlation of determination, or r-squared value

## limit_parameters.py
Functions:
- **limit_iterative_include(DataFrame, limit_num, log_progress=True)**: limits the input dimensionality of data found in *DataFrame* to a dimensionality of *limit_num* using a "retain the best" algorithm.
- **limit_genetic(DataFrame, limit_num, population_size, num_survivors, num_generations, num_processes, shuffle=False, data_split=[0.65, 0.25, 0.1], log_progress=True)**: limits the input dimensionality of data found in *DataFrame* to a dimensionality of *limit_num* using a genetic algorithm; *population_size* indicates the number of members for each generation, *num_survivors* indicates how many members of each generation survive, *num_generations* indicates how many generations the genetic algorithm runs for, *num_processes* specifies the number of parallel processes used for creating each generation, *shuffle* indicates whether to shuffle data set assignments for each population member, and *data_split* specifies the data set assignments if shuffle == True. Will log limiting progress if *log_progress* == True.
