# ECNet Examples

### Here are brief descriptions of the example scripts:

  - **config.yml**: example ECNet configuration file, set up for a cetane number prediction project
  - **create_project.py**: creating the project environment, importing data, training trial ANN's with validation, selecting the best performing trial for each build node, and publishing the project to a '.project' file
  - **use_project.py**: opening a published project, handing the trained model(s) a new database to predict values for, obtaining and saving the results, and calculating and listing various metrics of error/accuracy regarding the predicted values (if true values are known)
  - **limit_db_parameters.py**: imports a database, reduces the input dimensionality using a "retain the best" algorithm, and saves the reduced database to a specified file
  - **create_static_test_set.py**: Imports a dataset, and creates two files; one containing the test data, one containing the training (learning + validation) data; set sizes are determined by 'data_split' server variable
  - **select_from_test_set_performance.py**: Select best trial from each node using static test set performance
