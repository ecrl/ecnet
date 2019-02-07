from ecnet import Server


def main():

    sv = Server(log_level='debug', num_processes=4)
    sv.import_data(
        '../kv_model_v1.0.csv',
        sort_type='random',
        data_split=[0.6, 0.2, 0.2]
    )
    sv.create_project(
        'kinetic_viscosity',
        num_builds=1,
        num_nodes=5,
        num_candidates=25
    )
    sv.train_model(
        validate=True,
        shuffle='train',
        data_split=[0.6, 0.2, 0.2]
    )
    sv.select_best(dset='test')
    sv.save_project(clean_up=True)


if __name__ == '__main__':

    main()
