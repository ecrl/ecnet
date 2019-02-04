from ecnet import Server


def main():

    sv = Server(
        project_file='../kinetic_viscosity.prj',
        log_level='debug'
    )
    sv.use_model(
        dset='test',
        output_filename='../kv_test_results.csv'
    )
    sv.calc_error(
        'rmse',
        'mean_abs_error',
        'med_abs_error',
        'r2',
        dset='test'
    )


if __name__ == '__main__':

    main()
