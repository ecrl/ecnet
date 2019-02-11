from ecnet import Server
from ecnet.utils.logging import logger


def tune(num_processes, shuffle=False, split=[0.7, 0.2, 0.1], eval_set=None,
         eval_fn='rmse'):

    logger.stream_level = 'info'
    sv = Server(num_processes=num_processes)
    sv.load_data('cn_model_v1.0.csv', random=shuffle, split=split)
    sv.tune_hyperparameters(2, 2, shuffle=shuffle, split=split,
                            eval_set=eval_set, eval_fn=eval_fn)


if __name__ == '__main__':

    tune(1)
    tune(4)
    tune(1, True)
    tune(1, eval_set='test')
    tune(1, eval_set='test', eval_fn='mean_abs_error')
