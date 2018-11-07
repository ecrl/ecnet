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
    results = sv.use_model(dset)
    sv.save_results(results, 'test_results.csv')
    print(sv.calc_error('r2', 'rmse', 'mean_abs_error', 'med_abs_error'))
    sv.save_project()

if __name__ == '__main__':

    build(True, 'lv', 'test', 'random')
    build(True, 'lvt', 'test', 'random')
    build(False, 'lv', 'test', 'random')
    build(True, None, 'test', 'explicit')
    build(True, 'lv', 'learn', 'random')
    build(True, 'lv', 'valid', 'random')
    build(True, 'lv', 'train', 'random')
    build(True, 'lv', None, 'random')
