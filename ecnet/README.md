# Low-level usage of model, data_utils, error_utils, limit_parameters, and abc

## model.py
#### Class: MultilayerPerceptron
Attributes:
- **layers**: list of layers; layers contain information about the number of neurons and the activation function in form [num, func]
- **weights**: list of TensorFlow weight variables
- **biases**: list of TensorFlow bias variables

Methods:
- **add_layer(size, act_fn)**: appends a *Layer* to the MLP's layer list
  - supported activation functions: 'relu', 'sigmoid', 'linear'
- **connect_layers()**: initializes TensorFlow variables for weights and biases between each layer; fully connected
- **fit(x_l, y_l, learning_rate, train_epochs)**: fits the MLP to the inputs (**x_l**) and outputs (**y_l**) for **train_epochs** iterations with a learning rate of **learning_rate**
- **fit_validation(x_l, x_v, y_l, y_v, learning_rate, max_epochs)**: fits the MLP, periodically checking MLP performance using validation data; learning is stopped when validation data performance stops improving
  - **max_epochs** is the cutoff point if mdrmse has not fallen below mdrmse_stop
- **use(x)**: used to pass data through the trained model to get a prediction; returns predicted values
- **save(filepath)**: saves the TensorFlow session (.sess) and model architecture information (.struct) to specified filename
- **load(filepath)**: opens a TensorFlow session (.sess) and model architecture information (.struct) from specified filename

## data_utils.py
#### Class: DataFrame
Methods:
- **__init__(filename)**: imports a formatted database, creates DataPoints for each data entry, grabs string and group names and counts
- **create_sets(random = True, split = [0.7, 0.2, 0.1]**: create learning, validation and testing sets with *split* proportions; if random = False, database set assignments are used
- **create_sorted_sets(sort_string, split = [0.7, 0.2, 0.1]**: using *sort_string*, a string contained in the given database, assigns proportions *split* of each possible string value to learning, validation and testing sets
- **shuffle(args, split = [0.7, 0.2, 0.1])**: shuffles data for specified sets
   - args combinations:
    - 'l, v, t' (shuffles data for learning, validation and testing sets)
    - 'l, v' (shuffles data for learning and validation sets)
- **package_sets()** returns a PackagedData object, containing NumPy arrays for learning, validation and testing input and target sets

Functions:
- **output_results(results, DataFrame, filename)**: outputs *results* (calculated by model.py for a specified data set) to *filename*; *DataFrame* is required for outputting data entry names, strings, groups, etc.

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
- **limit_iterative_include(DataFrame, limit_num)**: limits the input dimensionality of data found in *DataFrame* to a dimensionality of *limit_num* using a "retain the best" algorithm
- **limit_genetic(DataFrame, limit_num, population_size, num_survivors, num_generations, print_feedback)**: limits the input dimensionality of data found in *DataFrame* to a dimensionality of *limit_num* using a genetic algorithm; *population_size* indicates the number of members for each generation, *num_survivors* indicates how many members of each generation survive, *num_generations* indicates how many generations the genetic algorithm runs for, and *print_feedback* is a boolean for the genetic algorithm to periodically print status updates
