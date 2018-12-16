from ecnet import Server


def limit_iterative(output_filename=None):

    sv = Server(log_level='debug')
    sv._logger.log('crit', 'LIMIT ITER | output file: {}'.format(
        num_processes
    ), call_loc={'call_loc': 'UNIT TESTING'})
    sv.import_data('cn_model_v1.0.csv')
    sv.limit_input_parameters(
        3, output_filename=output_filename
    )


def limit_genetic(shuffle=False, output_filename=None):

    sv = Server(log_level='debug')
    sv._logger.log('crit', 'LIMIT GEN | shuffle: {} | output file: {}'.format(
        shuffle, num_processes
    ), call_loc={'call_loc': 'UNIT TESTING'})
    sv.import_data('cn_model_v1.0.csv')
    sv.limit_input_parameters(
        3,
        use_genetic=True,
        population_size=3,
        num_generations=2,
        shuffle=shuffle,
        output_filename=output_filename
    )


if __name__ == '__main__':

    limit_iterative()
    limit_genetic()
    limit_genetic(shuffle=True, output_filename='limited_inputs.csv')
