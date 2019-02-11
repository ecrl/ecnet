from ecnet import Server
from ecnet.utils.logging import logger


def create_project():

    sv = Server()
    sv.load_data('cn_model_v1.0.csv')
    sv.create_project(
        'use_project',
        num_pools=2,
        num_candidates=2
    )
    sv.train()
    sv.save_project()


def use_project():

    sv = Server(prj_file='use_project.prj')
    sv.use('train', output_filename='use_project_train.csv')
    sv.use('test', output_filename='use_project_test.csv')


if __name__ == '__main__':

    create_project()
    use_project()
