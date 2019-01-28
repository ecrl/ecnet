from ecnet import Server


def tune(num_processes):

    sv = Server(log_level='debug', num_processes=num_processes)
    sv._logger.log('crit', 'TUNE HYPERPARAMS | processes: {}'.format(
        num_processes
    ), call_loc='UNIT TESTING')
    sv.import_data('cn_model_v1.0.csv')
    sv.tune_hyperparameters(
        num_employers=10,
        num_iterations=2
    )


if __name__ == '__main__':

    tune(1)
    tune(4)
