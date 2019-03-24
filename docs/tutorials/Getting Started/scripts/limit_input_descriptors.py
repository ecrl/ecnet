from ecnet import Server
from ecnet.utils.logging import logger


def main():

    logger.stream_level = 'debug'
    sv = Server(num_processes=4)
    sv.load_data('../kv_model_v1.0_full.csv')
    sv.limit_inputs(15, output_filename='../kv_model_v1.0.csv')


if __name__ == '__main__':

    main()
