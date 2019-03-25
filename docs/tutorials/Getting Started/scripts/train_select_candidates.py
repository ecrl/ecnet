from ecnet import Server
from ecnet.utils.logging import logger


def main():

    logger.stream_level = 'debug'
    sv = Server(num_processes=4)
    sv.load_data('../kv_model_v1.0.csv')
    sv.create_project(
        'kinetic_viscosity',
        num_pools=5,
        num_candidates=25
    )
    sv.train(
        shuffle='train',
        split=[0.7, 0.2, 0.1],
        validate=True,
        selection_set='test'
    )
    sv.save_project(del_candidates=True)


if __name__ == '__main__':

    main()
