from ecnet.tools.project import predict


def prj_predict():

    predict(
        'mols_smiles.txt',
        'cn_test_results.csv',
        'cn_model_v4.0.prj',
        form='smiles'
    )


if __name__ == '__main__':

    prj_predict()
