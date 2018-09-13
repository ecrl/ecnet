# Import the Server object
from ecnet import Server

if __name__ == '__main__':

    # Create Server
    sv = Server()

    # Create ECNet project
    sv.create_project('my_project')

    # Import data (change 'my_data.csv' to your database name)
    sv.import_data('my_data.csv')

    # Tune neural network hyperparemters (learning rate, maximum
    # training epochs during validation, number of neurons in
    # each hidden layer)
    hp = sv.tune_hyperparameters()

    # Print the tuned hyperparameters
    print(hp)

    # Tuned hyperparameters are now ready to be used!
    sv.train_model('shufflelv', validate=True)

    # Save ECNet project
    sv.save_project()
