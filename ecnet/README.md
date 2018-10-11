# Low-level usage of model, data_utils, error_utils, limit_parameters, and abc

## model.py
#### Class: MultilayerPerceptron

Methods:
- **add_layer(size, act_fn)**: appends a *Layer* to the MLP's layer list
  - supported activation functions: 'relu', 'sigmoid', 'linear', 'softmax'
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
- **limit_iterative_include(DataFrame, limit_num)**: limits the input dimensionality of data found in *DataFrame* to a dimensionality of *limit_num* using a "retain the best" algorithm.
- **limit_genetic(DataFrame, limit_num, population_size, num_survivors, num_generations, num_processes, shuffle=False, data_split=[0.65, 0.25, 0.1], logger=None)**: limits the input dimensionality of data found in *DataFrame* to a dimensionality of *limit_num* using a genetic algorithm; *population_size* indicates the number of members for each generation, *num_survivors* indicates how many members of each generation survive, *num_generations* indicates how many generations the genetic algorithm runs for, *num_processes* specifies the number of parallel processes used for creating each generation, *shuffle* indicates whether to shuffle data set assignments for each population member, and *data_split* specifies the data set assignments if shuffle == True. If a ColorLogger *logger* is not supplied, a new logger is initialized.
