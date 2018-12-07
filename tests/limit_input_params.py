from ecnet import Server


def limit_iterative():

    sv = Server()
    sv.import_data('cn_model_v1.0.csv', sort_type='random')
    sv.limit_input_parameters(3, 'test_limit_1.csv')


def limit_genetic():

    sv = Server()
    sv.import_data('cn_model_v1.0.csv', sort_type='random')
    sv.limit_input_parameters(
        3,
        'test_limit_2.csv',
        use_genetic=True,
        population_size=3,
        num_generations=2,
        shuffle=True,
        data_split=[0.65, 0.25, 0.1]
    )


def limit_genetic_mp():

    sv = Server(num_processes=4)
    sv.import_data('cn_model_v1.0.csv', sort_type='random')
    sv.limit_input_parameters(
        3,
        'test_limit_2.csv',
        use_genetic=True,
        population_size=3,
        num_generations=2,
        shuffle=True,
        data_split=[0.65, 0.25, 0.1]
    )

if __name__ == '__main__':

    limit_iterative()
    limit_genetic()
    limit_genetic_mp()
