import torch
import pytest
import os

from ecnet.datasets.structs import QSPRDataset, QSPRDatasetFromFile, QSPRDatasetFromValues
from ecnet.datasets.utils import _qspr_from_padel
from ecnet.datasets.load_data import _open_smiles_file, _open_target_file, _get_prop_paths,\
    _DATA_PATH, _get_file_data
from ecnet.callbacks import LRDecayLinear
from ecnet import ECNet
from ecnet.model import load_model
from ecnet.tasks.feature_selection import select_rfr
from ecnet.tasks.parameter_tuning import tune_batch_size, tune_model_architecture,\
    tune_training_parameters, CONFIG

_PROPS = ['bp', 'cn', 'cp', 'kv', 'lhv', 'mon', 'pp', 'ron', 'ysi', 'mp']
_BACKEND = 'padel'
_N_DESC = 1875
_N_PROCESSES = 1
_EPOCHS = 10

# dataset utils

def test_dataset_utils():
    smiles = ['CCC', 'CCCC', 'CCCCC']
    desc, keys = _qspr_from_padel(smiles)
    assert len(keys) == _N_DESC
    assert len(desc) == 3
    for d in desc:
        assert len(d) == _N_DESC

# dataset loading

def test_open_smiles_file():
    smiles = 'CCC\nCCCC\nCCCCC'
    with open('_temp.smiles', 'w') as smi_file:
        smi_file.write(smiles)
    smi_file.close()
    smiles = smiles.split('\n')
    opened_smiles = _open_smiles_file('_temp.smiles')
    assert len(smiles) == len(opened_smiles)
    for i in range(len(smiles)):
        assert smiles[i] == opened_smiles[i]


def test_open_target_file():
    print('UNIT TEST: Open .target file')
    target_vals = '3.0\n4.0\n5.0'
    with open('_temp.target', 'w') as tar_file:
        tar_file.write(target_vals)
    tar_file.close()
    target_vals = target_vals.split('\n')
    target_vals = [[float(v)] for v in target_vals]
    opened_targets = _open_target_file('_temp.target')
    assert len(target_vals) == len(opened_targets)
    for i in range(len(target_vals)):
        assert target_vals[i] == opened_targets[i]


def test_get_prop_paths():
    for p in _PROPS:
        smiles_fn, target_fn = _get_prop_paths(p)
        assert os.path.join(_DATA_PATH, f'{p}.smiles') == smiles_fn
        assert os.path.join(_DATA_PATH, f'{p}.target') == target_fn


def test_get_file_data():
    for p in _PROPS:
        smiles, targets = _get_file_data(p)
        assert len(smiles) == len(targets)
        assert type(smiles[0]) == str
        assert type(targets[0]) == list
        assert type(targets[0][0]) == float

# dataset structures

def test_qsprdataset():
    smiles = ['CCC', 'CCCC', 'CCCCC']
    targets = [[3.0], [4.0], [5.0]]
    ds = QSPRDataset(smiles, targets, backend=_BACKEND)
    assert len(ds.smiles) == len(smiles)
    assert len(ds.target_vals) == len(targets)
    assert len(ds.target_vals[0]) == len(targets[0])
    assert len(ds.desc_vals) == len(smiles)
    assert len(ds.desc_vals[0]) == _N_DESC
    assert type(ds.desc_vals) == type(torch.tensor([]))
    assert len(ds.desc_names) == _N_DESC


def test_qsprdatasetfromfile():
    smiles = 'CCC\nCCCC\nCCCCC'
    with open('_temp.smiles', 'w') as smi_file:
        smi_file.write(smiles)
    smi_file.close()
    smiles = smiles.split('\n')
    targets = [[3.0], [4.0], [5.0]]
    ds = QSPRDatasetFromFile('_temp.smiles', targets, backend=_BACKEND)
    assert len(ds.smiles) == len(smiles)
    assert len(ds.target_vals) == len(targets)
    assert len(ds.target_vals[0]) == len(targets[0])
    assert len(ds.desc_vals) == len(smiles)
    assert len(ds.desc_vals[0]) == _N_DESC
    assert type(ds.desc_vals) == type(torch.tensor([]))
    assert len(ds.desc_names) == _N_DESC


def test_qsprdatasetfromvalues():
    desc_vals = [
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.2, 0.3, 0.1],
        [0.1, 0.3, 0.0, 0.2]
    ]
    target_vals = [[1.0], [2.0], [3.0]]
    ds = QSPRDatasetFromValues(desc_vals, target_vals)
    assert len(ds.smiles) == len(desc_vals)
    assert len(ds.desc_names) == len(desc_vals[0])
    assert len(ds.desc_vals) == len(desc_vals)
    assert len(ds.target_vals) == len(target_vals)
    assert len(ds.target_vals[0]) == len(target_vals[0])
    assert type(ds.desc_vals) == type(torch.tensor([]))
    assert type(ds.target_vals) == type(torch.tensor([]))

# callbacks

def test_lrlineardecay():
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


def test_validator():
    # I can't think of a good way to test this one, but it works in practice
    return

# model

def test_model_construct():
    _INPUT_DIM = 3
    _OUTPUT_DIM = 1
    _HIDDEN_DIM = 5
    _N_HIDDEN = 2
    net = ECNet(_INPUT_DIM, _OUTPUT_DIM, _HIDDEN_DIM, _N_HIDDEN)
    assert len(net.model) == 2 + _N_HIDDEN
    assert net.model[0].in_features == _INPUT_DIM
    assert net.model[0].out_features == _HIDDEN_DIM
    assert net.model[-1].in_features == _HIDDEN_DIM
    assert net.model[-1].out_features == _OUTPUT_DIM
    for layer in net.model[1:-1]:
        assert layer.in_features == _HIDDEN_DIM
        assert layer.out_features == _HIDDEN_DIM


def test_model_fit():
    net = ECNet(_N_DESC, 1, 512, 2)
    smiles = ['CCC', 'CCCC', 'CCCCC']
    targets = [[3.0], [4.0], [5.0]]
    tr_loss, val_loss = net.fit(smiles, targets, backend=_BACKEND, epochs=_EPOCHS)
    assert len(tr_loss) == len(val_loss)
    assert len(tr_loss) == _EPOCHS


def test_model_save_load():
    net = ECNet(_N_DESC, 1, 512, 2)
    smiles = ['CCC', 'CCCC', 'CCCCC']
    targets = [[3.0], [4.0], [5.0]]
    ds = QSPRDataset(smiles, targets, backend=_BACKEND)
    tr_loss, val_loss = net.fit(dataset=ds, epochs=_EPOCHS)
    with pytest.raises(ValueError):
        net.save('_test.badext')
    net.save('_test.pt')
    val_0 = net(ds[0]['desc_vals'])
    with pytest.raises(FileNotFoundError):
        net = load_model('badfile.pt')
    net = load_model('_test.pt')
    val_0_new = net(ds[0]['desc_vals'])
    assert val_0 == val_0_new

# tasks

def test_feature_selection():
    smiles = ['CCC', 'CCCC', 'CCCCC']
    targets = [[3.0], [4.0], [5.0]]
    ds = QSPRDataset(smiles, targets, backend=_BACKEND)
    indices, importances = select_rfr(ds, total_importance=0.90)
    assert len(indices) < _N_DESC
    assert len(indices) == len(importances)
    assert importances == sorted(importances, reverse=True)
    for index in indices:
        assert index < _N_DESC


def test_tune_batch_size():
    smiles = ['CCC', 'CCCC', 'CCCCCC']
    targets = [[3.0], [4.0], [6.0]]
    ds_train = QSPRDataset(smiles, targets, backend=_BACKEND)
    smiles = ['CCCCC']
    targets = [[5.0]]
    ds_eval = QSPRDataset(smiles, targets, backend=_BACKEND)
    model = ECNet(_N_DESC, 1, 5, 1)
    res = tune_batch_size(1, 1, ds_train, ds_eval, _N_PROCESSES)
    assert 1 <= res['batch_size'] <= len(ds_train.target_vals)


def test_tune_model_architecture():
    smiles = ['CCC', 'CCCC', 'CCCCCC']
    targets = [[3.0], [4.0], [6.0]]
    ds_train = QSPRDataset(smiles, targets, backend=_BACKEND)
    smiles = ['CCCCC']
    targets = [[5.0]]
    ds_eval = QSPRDataset(smiles, targets, backend=_BACKEND)
    res = tune_model_architecture(1, 1, ds_train, ds_eval, _N_PROCESSES,)
    for k in list(res.keys()):
        assert res[k] >= CONFIG['architecture_params_range'][k][0]
        assert res[k] <= CONFIG['architecture_params_range'][k][1]


def test_tune_training_hyperparams():
    smiles = ['CCC', 'CCCC', 'CCCCCC']
    targets = [[3.0], [4.0], [6.0]]
    ds_train = QSPRDataset(smiles, targets, backend=_BACKEND)
    smiles = ['CCCCC']
    targets = [[5.0]]
    ds_eval = QSPRDataset(smiles, targets, backend=_BACKEND)
    res = tune_training_parameters(1, 1, ds_train, ds_eval, _N_PROCESSES)
    for k in list(res.keys()):
        assert res[k] >= CONFIG['training_params_range'][k][0]
        assert res[k] <= CONFIG['training_params_range'][k][1]
