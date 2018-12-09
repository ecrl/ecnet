from ecnet import Server


def build(validate, shuffle, dset, sort_type):

    sv = Server()
    sv.import_data('cn_model_v1.0.csv', sort_type=sort_type)
    sv.create_project(
        'test_project',
        num_builds=1,
        num_nodes=2,
        num_candidates=2
    )
    sv.train_model(validate=validate, shuffle=shuffle)
    sv.select_best(dset)
    sv.use_model(dset, output_filename='test_results.csv')
    print(sv.calc_error('r2', 'rmse', 'mean_abs_error', 'med_abs_error'))
    sv.save_project()


def build_mp():

    sv = Server(num_processes=4)
    sv.import_data('cn_model_v1.0.csv', sort_type='random')
    sv.create_project(
        'test_project_mp',
        num_builds=1,
        num_nodes=2,
        num_candidates=2
    )
    sv.train_model(validate=True, shuffle='train')
    sv.select_best('test')
    sv.use_model('test', output_filename='test_mp_results.csv')
    print(sv.calc_error('r2', 'rmse', 'mean_abs_error', 'med_abs_error'))
    sv.save_project()


def save_different_filename():

    sv = Server(num_processes=4)
    sv.import_data('cn_model_v1.0.csv', sort_type='random')
    sv.create_project(
        'test_save',
        num_builds=1,
        num_nodes=2,
        num_candidates=2
    )
    sv.train_model(validate=True, shuffle='train')
    sv.select_best(dset='test')
    sv.use_model(dset='test', output_filename='test_results.csv')
    print(sv.calc_error('r2', 'rmse', 'mean_abs_error', 'med_abs_error'))
    sv.save_project(filename='./newlocation.prj')


if __name__ == '__main__':

    build(True, 'train', 'test', 'random')
    build(True, 'all', 'test', 'random')
    build(False, 'train', 'test', 'random')
    build(True, None, 'test', 'explicit')
    build(True, 'train', 'learn', 'random')
    build(True, 'train', 'valid', 'random')
    build(True, 'train', 'train', 'random')
    build(True, 'train', None, 'random')
    build_mp()
    save_different_filename()
