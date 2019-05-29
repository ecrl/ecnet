from ecnet import Server
from ecnet.utils.logging import logger


def tune(num_processes, shuffle=None, split=[0.7, 0.2, 0.1], validate=True,
         eval_set=None, eval_fn='rmse'):

    logger.stream_level = 'debug'
    sv = Server(num_processes=num_processes)
    sv.load_data('cn_model_v1.0.csv', random=True, split=split)
    sv.tune_hyperparameters(2, 2, shuffle=shuffle, split=split,
                            validate=validate, eval_set=eval_set,
                            eval_fn=eval_fn)


if __name__ == '__main__':

    tune(1)
    tune(1, validate=False)
    tune(4)
    tune(1, 'all')
    tune(1, 'train')
    tune(1, eval_set='test')
    tune(1, eval_set='test', eval_fn='mean_abs_error')
