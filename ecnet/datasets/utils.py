r"""Utility functions for generating QSPR descriptors"""
from typing import List, Tuple
from alvadescpy import alvadesc, smiles_to_descriptors
from padelpy import from_smiles


def _qspr_from_padel(smiles: List[str], timeout: int = None) -> Tuple[List[List[float]], List[str]]:
    """
    Args:
        smiles (list[str]): list of SMILES strings
        timeout (int, optional): timeout for PaDEL-Descriptor process call; if None, uses
        max(15, len(smiles)) seconds; default = None

    Returns:
        Tuple[List[List[float]], List[str]]: (descriptors w/ shape (n_compounds, n_desc),
            descriptor names)
    """

    if timeout is None:
        timeout = len(smiles)
    desc = from_smiles(smiles, timeout=max(15, len(smiles)))
    keys = list(desc[0].keys())
    for idx, d in enumerate(desc):
        for k in keys:
            if d[k] == '':
                desc[idx][k] = 0.0
    desc = [[float(d[k]) for k in keys] for d in desc]
    return (desc, keys)


def _qspr_from_alvadesc(smiles: List[str]) -> Tuple[List[List[float]], List[str]]:
    """
    Args:
        smiles (list[str]): list of SMILES strings

    Returns:
        Tuple[List[List[float]], List[str]]: (descriptors w/ shape (n_compounds, n_desc),
            descriptor names)
    """

    desc = smiles_to_descriptors(smiles)
    keys = list(desc[0].keys())
    for idx, d in enumerate(desc):
        for k in keys:
            if d[k] == 'na':
                desc[idx][k] = 0.0
    desc = [[float(d[k]) for k in keys] for d in desc]
    return (desc, keys)


def _qspr_from_alvadesc_smifile(smiles_fn: str) -> Tuple[List[List[float]], List[str]]:
    """
    Args:
        smiles (list[str]): list of SMILES strings

    Returns:
        Tuple[List[List[float]], List[str]]: (descriptors w/ shape (n_compounds, n_desc),
            descriptor names)
    """

    desc = alvadesc(input_file=smiles_fn, inputtype='SMILES',
                    descriptors='ALL', labels=True)
    for d in desc:
        d.pop('No.')
        d.pop('NAME')
    keys = list(desc[0].keys())
    for idx, d in enumerate(desc):
        for k in keys:
            if d[k] == 'na':
                desc[idx][k] = 0.0
    desc = [[float(d[k]) for k in keys] for d in desc]
    return (desc, keys)
