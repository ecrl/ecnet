from ecnet import Server
from ecnet.utils.logging import logger


def main():

    logger.stream_level = 'debug'
    sv = Server(num_processes=4)
    sv.load_data('../kv_model_v1.0.csv')
    sv.tune_hyperparameters(20, 20, shuffle='train', split=[0.7, 0.2, 0.1],
                            eval_set='test')


if __name__ == '__main__':

    main()
