r"""
torch.nn.Module for regressing on target values given SMILES strings

Developed in 2021 by <Travis_Kessler@student.uml.edu>
"""

from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from re import compile

from .datasets.structs import QSPRDataset
from .callbacks import CallbackOperator, LRDecayLinear, Validator

_TORCH_MODEL_FN = compile(r'.*\.pt')


class ECNet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_hidden: int,
                 dropout: float = 0.0, device: str = 'cpu'):
        """
        ECNet, child of torch.nn.Module: handles data preprocessing, multilayer perceptron training,
        stores multilayer perceptron layers/weights for continued usage/saving

        Args:
            input_dim (int): dimensionality of input data
            output_dim (int): dimensionalit of output data
            hidden_dim (int): number of neurons in hidden layer(s)
            n_hidden (int): number of hidden layers between input and output
            dropout (float, optional): neuron dropout probability, default 0.0
            device (str, optional): device to run tensor ops on, default cpu
        """

        super(ECNet, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._n_hidden = n_hidden
        self._dropout = dropout
        self.model = nn.ModuleList()
        self._construct()
        self.to(torch.device(device))

    def _construct(self):
        """
        _construct: given supplied architecture params, construct multilayer perceptron
        """

        self.model = nn.ModuleList()
        self.model.append(nn.Linear(self._input_dim, self._hidden_dim))
        for _ in range(self._n_hidden):
            self.model.append(nn.Linear(self._hidden_dim, self._hidden_dim))
        self.model.append(nn.Linear(self._hidden_dim, self._output_dim))

    def fit(self, smiles: List[str] = None, target_vals: List[List[float]] = None,
            dataset: QSPRDataset = None, backend: str = 'padel', batch_size: int = 32,
            epochs: int = 100, lr_decay: float = 0.0, valid_size: float = 0.0,
            valid_eval_iter: int = 1, patience: int = 16, verbose: int = 0,
            random_state: int = None, shuffle: bool = False,
            **kwargs) -> Tuple[List[float], List[float]]:
        """
        fit: fits ECNet to either (1) SMILES and target values, or (2) a pre-loaded QSPRDataset;
        the training process utilizes the Adam optimization algorithm, MSE loss, ReLU activation
        functions between fully-connected layers, and optionally (1) a decaying learning rate, and
        (2) periodic validation during regression; periodic validation is used to determine when
        training ends (i.e. when a new minimum validation loss is not achieved after N epochs)

        Args:
            smiles (list[str], optional): if `dataset` not supplied, generates QSPR descriptors
                using these SMILES strings for use as input data
            target_vals (list[list[float]], optional): if `dataset` not supplied, this data is
                used for regression; should be shape (n_samples, n_targets)
            dataset (QSPRDataset, optional): pre-loaded dataset with descriptors + target values
            backend (str, optional): if using SMILES strings and target values, specifies backend
                software to use for QSPR generation; either 'padel' or 'alvadesc', default 'padel'
            batch_size (int, optional): training batch size; default = 32
            epochs (int, optional): number of training epochs; default = 100
            lr_decay (float, optional): linear rate of decay for learning rate; default = 0.0
            valid_size (float, optional): supply >0.0 to utilize periodic validation; value
                specifies proportion of supplied data to be used for validation
            valid_eval_iter (int, optional): validation set is evaluated every `this` epochs;
                default = 1 (evaluated every epoch)
            patience (int, optional): if new lowest validation loss not found after `this` many
                epochs, terminate training, set model parameters to those observed @ lowest
                validation loss
            verbose (int, optional): if > 0, will print every `this` epochs; default = 0
            random_state (int, optional): random_state used by sklearn.model_selection.
                train_test_split; default = None
            shuffle (bool, optional): if True, shuffles training/validation data between epochs;
                default = False; random_state should be None
            **kwargs: arguments accepted by torch.optim.Adam (i.e. learning rate, beta values)

        Returns:
            Tuple[List[float], List[Union[float, None]]]: (training losses, validation losses); if
                valid_size == 0.0, (training losses, [0, ..., 0])
        """

        # Data preparation
        if dataset is None:
            dataset = QSPRDataset(smiles, target_vals, backend)
        if valid_size > 0.0:
            index_train, index_valid = train_test_split(
                [i for i in range(len(dataset))], test_size=valid_size,
                random_state=random_state
            )
            dataloader_train = DataLoader(
                Subset(dataset, index_train), batch_size=batch_size, shuffle=True
            )
            dataloader_valid = DataLoader(
                Subset(dataset, index_valid), batch_size=len(index_valid), shuffle=True
            )
        else:
            dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Adam optimization algorithm
        optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)

        # Set up callbacks
        CBO = CallbackOperator()
        if 'lr' in kwargs:
            _lr = kwargs.get('lr')
            _lrdecay = LRDecayLinear(_lr, lr_decay, optimizer)
            CBO.add_cb(_lrdecay)
        if valid_size > 0.0:
            _validator = Validator(dataloader_valid, self, valid_eval_iter, patience)
            CBO.add_cb(_validator)

        train_losses, valid_losses = [], []
        # TRAIN BEGIN
        CBO.on_train_begin()
        for epoch in range(epochs):

            # EPOCH BEGIN
            if not CBO.on_epoch_begin(epoch):
                break

            if shuffle:
                index_train, index_valid = train_test_split(
                    [i for i in range(len(dataset))], test_size=valid_size,
                    random_state=random_state
                )
                dataloader_train = DataLoader(
                    Subset(dataset, index_train), batch_size=batch_size, shuffle=True
                )
                dataloader_valid = DataLoader(
                    Subset(dataset, index_valid), batch_size=len(index_valid), shuffle=True
                )

            train_loss = 0.0
            self.train()

            for b_idx, batch in enumerate(dataloader_train):

                # BATCH BEGIN
                if not CBO.on_batch_begin(b_idx):
                    break

                optimizer.zero_grad()
                pred = self(batch['desc_vals'])
                target = batch['target_val']

                # BATCH END, LOSS BEGIN
                if not CBO.on_batch_end(b_idx):
                    break
                if not CBO.on_loss_begin(b_idx):
                    break

                loss = self.loss(pred, target)
                loss.backward()

                # LOSS END, STEP BEGIN
                if not CBO.on_loss_end(b_idx):
                    break
                if not CBO.on_step_begin(b_idx):
                    break

                optimizer.step()
                train_loss += loss.detach().item() * len(batch['target_val'])

                # STEP END
                if not CBO.on_step_end(b_idx):
                    break

            # Determine epoch loss for training, validation data
            train_loss /= len(dataloader_train.dataset)
            if valid_size > 0.0:
                valid_loss = _validator._most_recent_loss
            else:
                valid_loss = 0.0
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # Print losses if verbose
            if verbose:
                if epoch % verbose == 0:
                    print('Epoch: {} | Train loss: {} | Valid loss: {}'.format(
                        epoch, train_loss, valid_loss
                    ))

            # EPOCH END
            if not CBO.on_epoch_end(epoch):
                break

        # TRAIN END
        CBO.on_train_end()
        return (train_losses, valid_losses)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation of data through multilayer perceptron

        Args:
            x (torch.tensor): input data to feed forward

        Returns:
            torch.tensor: output of final model layer
        """

        for i in range(len(self.model) - 1):
            x = self.model[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self._dropout, training=self.training)
        return self.model[-1](x)

    def loss(self, pred: torch.tensor, target: torch.tensor) -> torch.tensor:
        r"""
        Computes mean squared error between predicted values, target values

        Args:
            pred (torch.tensor): predicted values, shape (n_samples, n_features)
            target (torch.tensor): real values, shape (n_samples, n_features)

        Returns:
            torch.tensor: MSE loss, shape (*, 1)
        """

        return F.mse_loss(pred, target)

    def save(self, model_filename: str):
        """
        Saves the model for later use

        Args:
            model_filename (str): filename/path to save model
        """

        if _TORCH_MODEL_FN.match(model_filename) is None:
            raise ValueError('Models must be saved with a `.pt` extension')
        torch.save(self, model_filename)


def load_model(model_filename: str) -> ECNet:
    """
    Loads a model for use

    Args:
        model_filename (str): filename/path to load model from
    """

    model = torch.load(model_filename)
    model.eval()
    return model
