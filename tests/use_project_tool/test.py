from ecnet import Server
from ecnet.tools.project import predict


def prj_predict():

    sv = Server()
    sv.create_project('model', 1, 1)
    sv.load_data('cn_model_v2.0.csv', random=True, split=[0.7, 0.2, 0.1])
    sv.train(validate=True)
    sv.save_project()

    predict(
        'mols_smiles.txt',
        'cn_test_results.csv',
        'model.prj',
        form='smiles'
    )


if __name__ == '__main__':

    prj_predict()
