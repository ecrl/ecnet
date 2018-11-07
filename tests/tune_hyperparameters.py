from ecnet import Server


def tune():

    sv = Server()
    sv.import_data(
        'cn_model_v1.0.csv',
        sort_type='random',
        data_split=[0.65, 0.25, 0.1]
    )
    sv.tune_hyperparameters(num_iterations=2, num_employers=2)

if __name__ == '__main__':

    tune()
