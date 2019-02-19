from ecnet import Server
from ecnet.utils.logging import logger


def limit(num_processes, output_filename=None):

    logger.stream_level = 'info'
    sv = Server(num_processes=num_processes)
    sv.load_data('cn_model_v1.0.csv')
    sv.limit_inputs(3, output_filename=output_filename)


if __name__ == '__main__':

    limit(1)
    limit(4, output_filename='limited_inputs.csv')
