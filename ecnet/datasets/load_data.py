r"""Pre-bundled data interface"""
from typing import List, Tuple, Union
from os import path

from .structs import QSPRDatasetFromFile

_DATA_PATH = path.join(
    path.dirname(path.abspath(__file__)),
    'data'
)


def _open_smiles_file(smiles_fn: str) -> List[str]:
    r"""
    Args:
        smiles_fn (str): filename/path for SMILES file

    Returns:
        list[str]: [smiles_0, ..., smiles_N]
    """

    with open(smiles_fn, 'r') as smi_file:
        smiles = smi_file.readlines()
    smi_file.close()
    smiles = [s.replace('\n', '') for s in smiles]
    return smiles


def _open_target_file(target_fn: str) -> List[List[float]]:
    r"""
    Args:
        target_fn (str): filename/path for target values file

    Returns:
        list[list[float]]: lists of target values, in preparation for torch.tensor of shape
            (n_targets, 1)
    """

    with open(target_fn, 'r') as tar_file:
        target = tar_file.readlines()
    tar_file.close()
    target = [[float(t.replace('\n', ''))] for t in target]
    return target


def _get_prop_paths(prop: str) -> Tuple[str, str]:
    r"""
    Args:
        prop (str): any in ['bp', 'cn', 'cp', 'kv', 'lhv', 'mon', 'pp', 'ron', 'ysi']

    Returns:
        tuple[str, str]: (path to smiles file (str), path to targets file (str))
    """

    return (
        path.join(_DATA_PATH, '{}.smiles'.format(prop)),
        path.join(_DATA_PATH, '{}.target'.format(prop))
    )


def _get_file_data(prop: str) -> Tuple[List[str], List[List[float]]]:
    r"""
    Args:
        prop (str): any in ['bp', 'cn', 'cp', 'kv', 'lhv', 'mon', 'pp', 'ron', 'ysi']

    Returns:
        tuple[list[str], list[list[float]]]: (smiles, targets)
    """

    fn_smiles, fn_target = _get_prop_paths(prop)
    smiles = _open_smiles_file(fn_smiles)
    target = _open_target_file(fn_target)
    return (smiles, target)


def _load_set(prop: str, backend: str) -> QSPRDatasetFromFile:
    r"""
    Args:
        prop (str): any in ['bp', 'cn', 'cp', 'kv', 'lhv', 'mon', 'pp', 'ron', 'ysi']

    Returns:
        QSPRDatasetFromFile: loaded set
    """

    fn_smiles, fn_target = _get_prop_paths(prop)
    target_vals = _open_target_file(fn_target)
    return QSPRDatasetFromFile(fn_smiles, target_vals, backend)


def load_bp(as_dataset: bool = False, backend: str = 'padel') -> Union[
            Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('bp')
    return _load_set('bp', backend)


def load_cn(as_dataset: bool = False, backend: str = 'padel') -> Union[
            Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('cn')
    return _load_set('cn', backend)


def load_cp(as_dataset: bool = False, backend: str = 'padel') -> Union[
            Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('cp')
    return _load_set('cp', backend)


def load_kv(as_dataset: bool = False, backend: str = 'padel') -> Union[
            Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('kv')
    return _load_set('kv', backend)


def load_lhv(as_dataset: bool = False, backend: str = 'padel') -> Union[
             Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('lhv')
    return _load_set('lhv', backend)


def load_mon(as_dataset: bool = False, backend: str = 'padel') -> Union[
             Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('mon')
    return _load_set('mon', backend)


def load_pp(as_dataset: bool = False, backend: str = 'padel') -> Union[
            Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('pp')
    return _load_set('pp', backend)


def load_ron(as_dataset: bool = False, backend: str = 'padel') -> Union[
             Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('ron')
    return _load_set('ron', backend)


def load_ysi(as_dataset: bool = False, backend: str = 'padel') -> Union[
             Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]:
    r"""
    Args:
        as_dataset (bool, optional) if True, return QSPRDatasetFromFile object housing data;
            otherwise, return tuple of smiles and target values
        backend (str, optional): any in ['padel', 'alvadesc']

    Returns:
        Union[Tuple[List[str], List[List[float]]], QSPRDatasetFromFile]: either tuple of (smiles,
            target vals) or QSPRDatasetFromFile
    """

    if not as_dataset:
        return _get_file_data('ysi')
    return _load_set('ysi', backend)
