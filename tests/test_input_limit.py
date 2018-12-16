from ecnet import Server


def limit_iterative(output_filename=None):

    sv = Server(log_level='debug')
    sv._logger.log('crit', 'LIMIT ITER | output file: {}'.format(
        output_filename
    ), call_loc={'call_loc': 'UNIT TESTING'})
    sv.import_data('cn_model_v1.0.csv')
    sv.limit_input_parameters(
        3, output_filename=output_filename
    )


def limit_genetic(num_processes, shuffle=False, output_filename=None):

    sv = Server(log_level='debug', num_processes=num_processes)
    sv._logger.log(
        'crit',
        'LIMIT GEN | Processes: {} | shuffle: {} | output file: {}'.format(
            num_processes, shuffle, output_filename
        ), call_loc={'call_loc': 'UNIT TESTING'}
    )
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
    limit_genetic(1)
    limit_genetic(4, shuffle=True, output_filename='limited_inputs.csv')
