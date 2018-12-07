from ecnet import Server


def test_file_logging():

    sv = Server(log_dir='./logs')
    sv.import_data('cn_model_v1.0.csv')
    sv.tune_hyperparameters(num_iterations=2, num_employers=2)
    sv.limit_input_parameters(
        1,
        'test.csv',
        use_genetic=True,
        population_size=2,
        num_generations=2
    )


if __name__ == '__main__':

    test_file_logging()
