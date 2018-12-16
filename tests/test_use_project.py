from ecnet import Server


def create_project():

    sv = Server(log_level='debug')
    sv._logger.log(
        'crit',
        'CREATING PROJECT',
        call_loc={'call_loc': 'UNIT TESTING'}
    )
    sv.import_data('cn_model_v1.0.csv')
    sv.create_project(
        'use_project',
        num_builds=1,
        num_nodes=2,
        num_candidates=3
    )
    sv.train_model()
    sv.select_best()
    sv.save_project(clean_up=True)


def use_project():

    sv = Server(project_file='use_project.prj')
    sv._logger.log(
        'crit',
        'USING PROJECT',
        call_loc={'call_loc': 'UNIT TESTING'}
    )
    sv.use_model('train', output_filename='use_project_train.csv')
    sv.use_model('test', output_filename='use_project_test.csv')


if __name__ == '__main__':

    create_project()
    use_project()
