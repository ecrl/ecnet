from ecnet import Server
from ecnet.utils.logging import logger


def main():

    logger.stream_level = 'debug'
    sv = Server()
    sv.load_data('cn_model_v1.0.csv')
    sv.remove_outliers(output_filename='cn_no_outliers.csv')


if __name__ == '__main__':

    main()
