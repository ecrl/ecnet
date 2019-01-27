from ecnet import Server


def train(validate, dset=None):

    sv = Server(log_level='debug')
    sv._logger.log('crit', 'NO PROJECT | validate: {} | dset: {}'.format(
        validate, dset
    ), call_loc='UNIT TESTING')
    sv.import_data('cn_model_v1.0.csv')
    sv.train_model(validate=validate)
    sv.use_model()
    sv.calc_error('rmse', 'med_abs_error', 'mean_abs_error', 'r2')


def train_project(validate, shuffle, num_processes, dset,
                  output_filename=None):

    sv = Server(num_processes=num_processes, log_level='debug')
    sv._logger.log(
        'crit',
        'PROJECT | validate: {} | shuffle: {} | processes: {} | dset: {} | file: {}'
        .format(
            validate, shuffle, num_processes, dset, output_filename
        ), call_loc='UNIT TESTING'
    )
    sv.import_data('cn_model_v1.0.csv')
    sv.create_project(
        'test_training_{}-{}-{}-{}'.format(
            validate, shuffle, num_processes, output_filename
        ),
        num_builds=1,
        num_nodes=5,
        num_candidates=10
    )
    sv.train_model(validate=validate, shuffle=shuffle)
    sv.select_best(dset=dset)
    sv.use_model(dset=dset, output_filename=output_filename)
    sv.calc_error('rmse', 'med_abs_error', 'mean_abs_error', 'r2')
    sv.save_project(clean_up=True)


if __name__ == '__main__':

    train(False)
    train(True)
    train(True, dset='learn')
    train(True, dset='valid')
    train(True, dset='train')
    train(True, dset='test')
    train_project(False, None, 1, 'test')
    train_project(True, 'train', 1, 'learn')
    train_project(True, 'all', 1, 'valid')
    train_project(True, 'train', 1, 'train')
    train_project(True, 'train', 1, 'test')
    train_project(True, 'train', 1, None)
    train_project(True, 'train', 4, 'test')
    train_project(True, 'train', 4, 'test', output_filename='test_results.csv')
