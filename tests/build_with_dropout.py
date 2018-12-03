from ecnet import Server


def build():

    sv = Server()
    sv.import_data('cn_model_v1.0.csv')
    sv.vars['keep_prob'] = 0.8
    sv.train_model(validate=True)
    results = sv.use_model('test')
    print(sv.calc_error('rmse', dset='test'))

if __name__ == '__main__':

    build()
