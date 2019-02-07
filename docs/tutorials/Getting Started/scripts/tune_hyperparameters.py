from ecnet import Server


def main():

    sv = Server(log_level='debug', num_processes=4)
    sv.import_data(
        '../kv_model_v1.0.csv',
        sort_type='random',
        data_split=[0.6, 0.2, 0.2]
    )
    sv.tune_hyperparameters(
        num_employers=30,
        num_iterations=10
    )


if __name__ == '__main__':

    main()
