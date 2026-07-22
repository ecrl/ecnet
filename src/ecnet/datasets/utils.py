r"""Utility functions for generating QSPR descriptors"""

from typing import List, Tuple


def _import_padelpy():
    """Import padelpy on demand (default backend)."""
    from padelpy import from_smiles

    return from_smiles


def _import_alvadescpy():
    """Import alvadescpy on demand (optional licensed backend)."""
    try:
        from alvadescpy import alvadesc, smiles_to_descriptors
    except ModuleNotFoundError as exc:
        # setuptools >=82 may omit pkg_resources; alvadescpy still imports it.
        if exc.name in {"pkg_resources", "alvadescpy"}:
            raise ModuleNotFoundError(
                "alvaDesc backend requires a working alvadescpy install. "
                "If the error mentions pkg_resources, install an older "
                "setuptools (for example `pip install 'setuptools<82'`) or "
                "use backend='padel'."
            ) from exc
        raise
    return alvadesc, smiles_to_descriptors


def _qspr_from_padel(
    smiles: List[str], timeout: int = None
) -> Tuple[List[List[float]], List[str]]:
    """
    Args:
        smiles (list[str]): list of SMILES strings
        timeout (int, optional): timeout for PaDEL-Descriptor process call; if None, uses
        max(15, len(smiles)) seconds; default = None

    Returns:
        Tuple[List[List[float]], List[str]]: (descriptors w/ shape (n_compounds, n_desc),
            descriptor names)
    """

    from_smiles = _import_padelpy()
    if timeout is None:
        timeout = len(smiles)
    desc = from_smiles(smiles, timeout=max(15, len(smiles)))
    keys = list(desc[0].keys())
    for idx, d in enumerate(desc):
        for k in keys:
            if d[k] == "":
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

    _, smiles_to_descriptors = _import_alvadescpy()
    desc = smiles_to_descriptors(smiles)
    keys = list(desc[0].keys())
    for idx, d in enumerate(desc):
        for k in keys:
            if d[k] == "na" or d[k] == r"na\r":
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

    alvadesc, _ = _import_alvadescpy()
    desc = alvadesc(
        input_file=smiles_fn, inputtype="SMILES", descriptors="ALL", labels=True
    )
    for d in desc:
        d.pop("No.")
        d.pop("NAME")
    keys = list(desc[0].keys())
    for idx, d in enumerate(desc):
        for k in keys:
            if d[k] == "na" or d[k] == r"na\r":
                desc[idx][k] = 0.0
    desc = [[float(d[k]) for k in keys] for d in desc]
    return (desc, keys)
