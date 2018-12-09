from ecnet import Server


def test_file_logging():

    sv = Server(log_dir='./logs', num_processes=4)
    sv.import_data('cn_model_v1.0.csv')
    sv.tune_hyperparameters(num_iterations=2, num_employers=2)
    sv.limit_input_parameters(
        1,
        'test.csv',
        use_genetic=True,
        population_size=2,
        num_generations=2
    )


def test_log_level(level):
    sv = Server(log_level=level)
    sv.import_data('cn_model_v1.0.csv')
    sv.create_project(
        'log_level_test',
        num_builds=1,
        num_nodes=2,
        num_candidates=2
    )
    sv.train_model()
    sv.save_project()


def test_level_setting():
    sv = Server()
    sv.import_data('cn_model_v1.0.csv')
    sv.tune_hyperparameters(num_iterations=2, num_employers=2)
    sv.log_level = 'debug'
    sv.tune_hyperparameters(num_iterations=2, num_employers=2)


if __name__ == '__main__':

    test_file_logging()
    test_log_level('debug')
    test_level_setting()
