from ecnet import Server
from ecnet.utils.logging import logger


def train(validate, dset=None):

    sv = Server()
    sv.load_data('cn_model_v1.0.csv')
    sv.train(validate=validate, selection_set=dset)
    sv.use(dset=dset)
    sv.errors('rmse', 'med_abs_error', 'mean_abs_error', 'r2', dset=dset)


def train_project(validate, shuffle, split=[0.7, 0.2, 0.1], num_processes=1,
                  dset=None, sel_fn='rmse', output_filename=None):

    sv = Server(num_processes=num_processes)
    sv.load_data('cn_model_v1.0.csv')
    sv.create_project(
        '_training_test',
        num_pools=2,
        num_candidates=2
    )
    sv.train(shuffle=shuffle, split=split, selection_set=dset,
             selection_fn=sel_fn, validate=validate)
    sv.use(dset=dset, output_filename=output_filename)
    sv.errors('rmse', 'med_abs_error', 'mean_abs_error', 'r2')
    sv.save_project()


def retrain():

    sv = Server(prj_file='_training_test.prj')
    sv.load_data('cn_model_v1.0.csv')
    sv.train(retrain=True)
    sv.save_project(del_candidates=True)


if __name__ == '__main__':

    logger.stream_level = 'debug'
    train(False)
    train(True)
    train(True, dset='learn')
    train(True, dset='valid')
    train(True, dset='train')
    train(True, dset='test')
    train_project(False, None, None, 1, None)
    train_project(True, 'all')
    train_project(True, 'all', num_processes=4)
    train_project(True, 'train', dset='test',
                  output_filename='_test_train_res.csv')
    train_project(True, 'train', dset='test', sel_fn='mean_abs_error')
    train_project(True, 'train', dset='test', sel_fn='med_abs_error')
    retrain()
