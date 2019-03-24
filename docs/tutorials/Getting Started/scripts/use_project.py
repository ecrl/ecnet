from ecnet import Server
from ecnet.utils.logging import logger


def main():

    logger.stream_level = 'debug'
    sv = Server(prj_file='kinetic_viscosity.prj')
    sv.use(dset='test', output_filename='../kv_test_results.csv')
    sv.errors('rmse', 'mean_abs_error', 'med_abs_error', 'r2', dset='test')


if __name__ == '__main__':

    main()
