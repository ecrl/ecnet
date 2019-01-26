from ecnet.tools import predict_with_prj


def predict():

    predict_with_prj(
        'mols_smiles.txt',
        '_test_results.csv',
        'cp_model_v1.5.prj',
        form='smiles'
    )


if __name__ == '__main__':

    predict()
