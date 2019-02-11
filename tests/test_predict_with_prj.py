from ecnet.tools.project import predict


def predict():

    predict(
        'mols_smiles.txt',
        'cp_test_results.csv',
        'cp_model_v1.5.prj',
        form='smiles'
    )


if __name__ == '__main__':

    predict()
