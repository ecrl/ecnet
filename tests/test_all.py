import torch
import unittest
import os

from ecnet.datasets.structs import QSPRDataset, QSPRDatasetFromFile, QSPRDatasetFromValues
from ecnet.datasets.utils import _qspr_from_padel, _qspr_from_alvadesc
from ecnet.datasets.load_data import _open_smiles_file, _open_target_file, _get_prop_paths,\
    _DATA_PATH, _get_file_data, _load_set
from ecnet.callbacks import LRDecayLinear, Validator
from ecnet import ECNet
from ecnet.tasks.feature_selection import select_rfr
from ecnet.tasks.parameter_tuning import tune_batch_size, tune_model_architecture,\
    tune_training_parameters, CONFIG

_PROPS = ['bp', 'cn', 'cp', 'kv', 'lhv', 'mon', 'pp', 'ron', 'ysi']
_BACKEND = 'padel'
_N_DESC = 1875
_N_PROCESSES = 4


class TestDatasetUtils(unittest.TestCase):

    def test_qspr_generation(self):

        smiles = ['CCC', 'CCCC', 'CCCCC']
        desc, keys = _qspr_from_padel(smiles)
        self.assertEqual(len(keys), 1875)
        self.assertEqual(len(desc), 3)
        for d in desc:
            self.assertEqual(len(d), 1875)


class TestDatasetLoading(unittest.TestCase):

    def test_open_smiles_file(self):

        smiles = 'CCC\nCCCC\nCCCCC'
        with open('_temp.smiles', 'w') as smi_file:
            smi_file.write(smiles)
        smi_file.close()
        smiles = smiles.split('\n')
        opened_smiles = _open_smiles_file('_temp.smiles')
        self.assertEqual(len(smiles), len(opened_smiles))
        for i in range(len(smiles)):
            self.assertEqual(smiles[i], opened_smiles[i])

    def test_open_target_file(self):

        target_vals = '3.0\n4.0\n5.0'
        with open('_temp.target', 'w') as tar_file:
            tar_file.write(target_vals)
        tar_file.close()
        target_vals = target_vals.split('\n')
        target_vals = [[float(v)] for v in target_vals]
        opened_targets = _open_target_file('_temp.target')
        self.assertEqual(len(target_vals), len(opened_targets))
        for i in range(len(target_vals)):
            self.assertEqual(target_vals[i], opened_targets[i])

    def test_get_prop_paths(self):

        for p in _PROPS:
            smiles_fn, target_fn = _get_prop_paths(p)
            self.assertEqual(
                os.path.join(_DATA_PATH, '{}.smiles'.format(p)), smiles_fn
            )
            self.assertEqual(
                os.path.join(_DATA_PATH, '{}.target'.format(p)), target_fn
            )

    def test_get_file_data(self):

        for p in _PROPS:
            smiles, targets = _get_file_data(p)
            self.assertEqual(len(smiles), len(targets))
            self.assertTrue(type(smiles[0]) == str)
            self.assertTrue(type(targets[0]) == list)
            self.assertTrue(type(targets[0][0]) == float)

    def tearDown(self):

        if os.path.exists('_temp.smiles'):
            os.remove('_temp.smiles')
        if os.path.exists('_temp.target'):
            os.remove('_temp.target')


class TestDatasetStructs(unittest.TestCase):

    def test_qsprdataset(self):

        smiles = ['CCC', 'CCCC', 'CCCCC']
        targets = [[3.0], [4.0], [5.0]]
        ds = QSPRDataset(smiles, targets, backend=_BACKEND)
        self.assertEqual(len(ds.smiles), 3)
        self.assertEqual(len(ds.target_vals), 3)
        self.assertEqual(len(ds.target_vals[0]), len(targets[0]))
        self.assertEqual(len(ds.desc_vals), 3)
        self.assertEqual(len(ds.desc_vals[0]), _N_DESC)
        self.assertEqual(type(ds.desc_vals), type(torch.tensor([])))
        self.assertEqual(len(ds.desc_names), _N_DESC)

    def test_qsprdatasetfromfile(self):

        smiles = 'CCC\nCCCC\nCCCCC'
        with open('_temp.smiles', 'w') as smi_file:
            smi_file.write(smiles)
        smi_file.close()
        smiles = smiles.split('\n')
        targets = [[3.0], [4.0], [5.0]]
        ds = QSPRDatasetFromFile('_temp.smiles', targets, backend=_BACKEND)
        self.assertEqual(len(ds.smiles), 3)
        self.assertEqual(len(ds.target_vals), 3)
        self.assertEqual(len(ds.target_vals[0]), len(targets[0]))
        self.assertEqual(len(ds.desc_vals), 3)
        self.assertEqual(len(ds.desc_vals[0]), _N_DESC)
        self.assertEqual(type(ds.desc_vals), type(torch.tensor([])))
        self.assertEqual(len(ds.desc_names), _N_DESC)

    def test_qsprdatasetfromvalues(self):

        desc_vals = [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.2, 0.3, 0.1],
            [0.1, 0.3, 0.0, 0.2]
        ]
        target_vals = [[1.0], [2.0], [3.0]]
        ds = QSPRDatasetFromValues(desc_vals, target_vals)
        self.assertEqual(len(ds.smiles), len(desc_vals))
        self.assertEqual(len(ds.desc_names), len(desc_vals[0]))
        self.assertEqual(len(ds.desc_vals), len(desc_vals))
        self.assertEqual(len(ds.target_vals), len(target_vals))
        self.assertEqual(len(ds.target_vals[0]), len(target_vals[0]))
        self.assertEqual(type(ds.desc_vals), type(torch.tensor([])))
        self.assertEqual(type(ds.target_vals), type(torch.tensor([])))

    def tearDown(self):

        if os.path.exists('_temp.smiles'):
            os.remove('_temp.smiles')


class TestCallbacks(unittest.TestCase):

    def test_lrlineardecay(self):

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        lr = 0.001
        lrd = 0.00001
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        linear_decay = LRDecayLinear(lr, lrd, optim)
        reached_epoch = 0
        for epoch in range(10000):
            if not linear_decay.on_epoch_begin(epoch):
                break
            reached_epoch += 1
        if reached_epoch > int(lr / lrd):
            raise RuntimeError('Linear decay: epoch reached {}'.format(reached_epoch))

    def test_validator(self):

        # I can't think of a good way to test this one, but it works in practice
        return


class TestModel(unittest.TestCase):

    def test_construct(self):

        _INPUT_DIM = 3
        _OUTPUT_DIM = 1
        _HIDDEN_DIM = 5
        _N_HIDDEN = 2
        net = ECNet(_INPUT_DIM, _OUTPUT_DIM, _HIDDEN_DIM, _N_HIDDEN)
        self.assertEqual(len(net.model), 2 + _N_HIDDEN)
        self.assertEqual(net.model[0].in_features, _INPUT_DIM)
        self.assertEqual(net.model[0].out_features, _HIDDEN_DIM)
        self.assertEqual(net.model[-1].in_features, _HIDDEN_DIM)
        self.assertEqual(net.model[-1].out_features, _OUTPUT_DIM)
        for layer in net.model[1:-1]:
            self.assertEqual(layer.in_features, _HIDDEN_DIM)
            self.assertEqual(layer.out_features, _HIDDEN_DIM)

    def test_fit(self):

        _EPOCHS = 10
        net = ECNet(_N_DESC, 1, 512, 2)
        smiles = ['CCC', 'CCCC', 'CCCCC']
        targets = [[3.0], [4.0], [5.0]]
        tr_loss, val_loss = net.fit(smiles, targets, backend=_BACKEND, epochs=_EPOCHS)
        self.assertEqual(len(tr_loss), len(val_loss))
        self.assertEqual(len(tr_loss), _EPOCHS)


class TestTasks(unittest.TestCase):

    def test_feature_selection(self):

        smiles = ['CCC', 'CCCC', 'CCCCC']
        targets = [[3.0], [4.0], [5.0]]
        ds = QSPRDataset(smiles, targets, backend=_BACKEND)
        indices, importances = select_rfr(ds, total_importance=0.90)
        self.assertTrue(len(indices) < _N_DESC)
        self.assertEqual(len(indices), len(importances))
        self.assertEqual(importances, sorted(importances, reverse=True))
        for index in indices:
            self.assertTrue(index < _N_DESC)

    def test_tune_batch_size(self):

        smiles = ['CCC', 'CCCC', 'CCCCCC']
        targets = [[3.0], [4.0], [6.0]]
        ds_train = QSPRDataset(smiles, targets, backend=_BACKEND)
        smiles = ['CCCCC']
        targets = [[5.0]]
        ds_eval = QSPRDataset(smiles, targets, backend=_BACKEND)
        model = ECNet(_N_DESC, 1, 5, 1)
        res = tune_batch_size(5, 5, _N_PROCESSES, model=model, train_ds=ds_train, eval_ds=ds_eval)
        self.assertTrue(1 <= res['batch_size'] <= len(ds_train.target_vals))

    def test_tune_model_architecture(self):

        smiles = ['CCC', 'CCCC', 'CCCCCC']
        targets = [[3.0], [4.0], [6.0]]
        ds_train = QSPRDataset(smiles, targets, backend=_BACKEND)
        smiles = ['CCCCC']
        targets = [[5.0]]
        ds_eval = QSPRDataset(smiles, targets, backend=_BACKEND)
        model = ECNet(_N_DESC, 1, 5, 1)
        res = tune_model_architecture(5, 5, _N_PROCESSES, model=model, train_ds=ds_train,
                                      eval_ds=ds_eval)
        for k in list(res.keys()):
            self.assertTrue(res[k] >= CONFIG['architecture_params_range'][k][0])
            self.assertTrue(res[k] <= CONFIG['architecture_params_range'][k][1])

    def test_tune_training_hps(self):

        smiles = ['CCC', 'CCCC', 'CCCCCC']
        targets = [[3.0], [4.0], [6.0]]
        ds_train = QSPRDataset(smiles, targets, backend=_BACKEND)
        smiles = ['CCCCC']
        targets = [[5.0]]
        ds_eval = QSPRDataset(smiles, targets, backend=_BACKEND)
        model = ECNet(_N_DESC, 1, 5, 1)
        res = tune_training_parameters(5, 5, _N_PROCESSES, model=model, train_ds=ds_train,
                                       eval_ds=ds_eval)
        for k in list(res.keys()):
            if k == 'betas':
                continue
            self.assertTrue(res[k] >= CONFIG['training_params_range'][k][0])
            self.assertTrue(res[k] <= CONFIG['training_params_range'][k][1])
        self.assertTrue(res['betas'][0] >= CONFIG['training_params_range']['beta_1'][0])
        self.assertTrue(res['betas'][0] <= CONFIG['training_params_range']['beta_1'][1])
        self.assertTrue(res['betas'][1] >= CONFIG['training_params_range']['beta_2'][0])
        self.assertTrue(res['betas'][1] <= CONFIG['training_params_range']['beta_2'][1])


if __name__ == '__main__':

    unittest.main()
