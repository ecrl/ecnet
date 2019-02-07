from ecnet import Server


def main():

    sv = Server(log_level='debug', num_processes=4)
    sv.import_data(
        '../kv_model_v1.0_full.csv',
        sort_type='random',
        data_split=[0.6, 0.2, 0.2]
    )
    sv.limit_input_parameters(
        limit_num=15,
        output_filename='../kv_model_v1.0.csv',
        use_genetic=True,
        population_size=30,
        num_generations=10,
        mut_rate=0.2,
        max_mut_amt=1
    )


if __name__ == '__main__':

    main()
