r"""PyTorch-iterable/callable data structures"""
from typing import List, Tuple, Iterable
import torch
from torch.utils.data import Dataset

from .utils import _qspr_from_padel, _qspr_from_alvadesc,\
    _qspr_from_alvadesc_smifile


class QSPRDataset(Dataset):

    def __init__(self, smiles: List[str], target_vals: Iterable[Iterable[float]],
                 backend: str = 'padel'):
        """
        QSPRDataset: creates a torch.utils.data.Dataset from SMILES strings and target values

        Args:
            smiles (list[str]): SMILES strings
            target_vals (Iterable[Iterable[float]]): target values of shape (n_samples, n_targets)
            backend (str, optional): backend for QSPR generation, ['padel', 'alvadesc']
        """

        self.smiles = smiles
        self.target_vals = torch.as_tensor(target_vals)
        self.desc_vals, self.desc_names = self.smi_to_qspr(smiles, backend)
        self.desc_vals = torch.as_tensor(self.desc_vals)

    @staticmethod
    def smi_to_qspr(smiles: List[str], backend: str) -> Tuple[List[List[float]], List[str]]:
        """
        Generate QSPR descriptors for each supplied SMILES string

        Args:
            smiles (list[str]): SMILES strings
            backend (str): backend for QSPR generation, ['padel', 'alvadesc']

        Returns:
            tuple[list[list[float]], list[str]]
        """

        if backend == 'padel':
            return _qspr_from_padel(smiles)
        elif backend == 'alvadesc':
            return _qspr_from_alvadesc(smiles)
        else:
            raise ValueError('Unknown backend software: {}'.format(backend))

    def set_index(self, index: List[int]):
        """
        Reduce the number of samples in the dataset; samples retained given by supplied indices

        Args:
            index (list[int]): indices of the dataset to retain, all others are removed
        """

        self.smiles = [self.smiles[i] for i in index]
        self.target_vals = torch.as_tensor([self.target_vals[i].numpy() for i in index])
        self.desc_vals = torch.as_tensor(
            [self.desc_vals[i].numpy() for i in index]
        )

    def set_desc_index(self, index: List[int]):
        """
        Reduce the number of features per sample; features retained given by supplied indices

        Args:
            index (list[int]): indices of the features to retain, all others are removed
        """

        self.desc_vals = torch.as_tensor(
            [[val[i] for i in index] for val in self.desc_vals]
        )
        self.desc_names = [self.desc_names[i] for i in index]

    def __len__(self):

        return len(self.smiles)

    def __getitem__(self, idx: int):
        """
        Dictionary representation of compound at index `idx`

        Args:
            idx (int): compound to return
        """

        smiles = self.smiles[idx]
        target_val = self.target_vals[idx]
        dv = self.desc_vals[idx]
        return {
            'smiles': smiles,
            'target_val': target_val,
            'desc_vals': dv,
            'desc_names': self.desc_names
        }


class QSPRDatasetFromFile(QSPRDataset):

    def __init__(self, smiles_fn: str, target_vals: Iterable[Iterable[float]],
                 backend: str = 'padel'):
        """
        QSPRDatasetFromFile: creates a torch.utils.data.Dataset given target values and a supplied
        filename/path to a SMILES file

        Args:
            smiles_fn (str): filename/path of SMILES file
            target_vals (Iterable[Iterable[float]]): target values of shape (n_samples, n_targets)
            backend (str, optional): backend for QSPR generation, ['padel', 'alvadesc']
        """

        self.smiles = self._open_smiles_file(smiles_fn)
        self.target_vals = torch.as_tensor(target_vals)
        if backend == 'padel':
            self.desc_vals, self.desc_names = self.smi_to_qspr(
                self.smiles, backend
            )
            self.desc_vals = torch.as_tensor(self.desc_vals)
        elif backend == 'alvadesc':
            self.desc_vals, self.desc_names = _qspr_from_alvadesc_smifile(
                smiles_fn
            )
            self.desc_vals = torch.as_tensor(self.desc_vals)

    @staticmethod
    def _open_smiles_file(smiles_fn: str) -> List[str]:
        """
        Open SMILES file at specified location

        Args:
            smiles_fn (str): filename/path of SMILES file

        Returns:
            list[str]: SMILES strings
        """

        with open(smiles_fn, 'r') as smi_file:
            smiles = smi_file.readlines()
        smi_file.close()
        smiles = [s.replace('\n', '') for s in smiles]
        return smiles


class QSPRDatasetFromValues(QSPRDataset):

    def __init__(self, desc_vals: Iterable[Iterable[float]],
                 target_vals: Iterable[Iterable[float]]):
        """
        QSPRDatasetFromValues: creates a torch.utils.data.Dataset given supplied descriptor values,
        supplied target values

        Args:
            desc_vals (Iterable[Iterable[float]]): descriptor values, shape (n_samples, n_features)
            target_vals (Iterable[Iterable[float]]): target values, shape (n_samples, n_targets)
        """

        self.smiles = ['' for _ in range(len(target_vals))]
        self.desc_names = ['' for _ in range(len(desc_vals[0]))]
        self.desc_vals = torch.as_tensor(desc_vals)
        self.target_vals = torch.as_tensor(target_vals)
