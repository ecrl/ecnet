r"""
torch.nn.Module for regressing on target values given SMILES strings

Developed in 2021 by <Travis_Kessler@student.uml.edu>
"""

from re import compile
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .callbacks import CallbackOperator, LRDecayLinear, Validator
from .datasets.structs import QSPRDataset

_TORCH_MODEL_FN = compile(r".*\.pt")
_STATE_FORMAT = "ecnet-state-v1"


class ECNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden: int,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
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

    def fit(
        self,
        smiles: List[str] = None,
        target_vals: List[List[float]] = None,
        dataset: QSPRDataset = None,
        backend: str = "padel",
        batch_size: int = 32,
        epochs: int = 100,
        lr_decay: float = 0.0,
        valid_size: float = 0.0,
        valid_eval_iter: int = 1,
        patience: int = 16,
        verbose: int = 0,
        random_state: int = None,
        shuffle: bool = False,
        **kwargs,
    ) -> Tuple[List[float], List[float]]:
        """
        Fit ECNet to SMILES/target values or a pre-loaded QSPRDataset.

        Training uses Adam, MSE loss, and ReLU activations between layers.
        Optional linear learning-rate decay and validation-based early stopping
        are supported when ``valid_size > 0``.

        Parameters
        ----------
        smiles : list[str], optional
            SMILES strings used to build descriptors when ``dataset`` is omitted.
        target_vals : list[list[float]], optional
            Regression targets when ``dataset`` is omitted.
        dataset : QSPRDataset, optional
            Pre-loaded dataset with descriptors and targets.
        backend : str, optional
            Descriptor backend when building from SMILES (``padel`` or
            ``alvadesc``). Default ``padel``.
        batch_size : int, optional
            Training batch size. Default 32.
        epochs : int, optional
            Number of training epochs. Default 100.
        lr_decay : float, optional
            Linear learning-rate decay per epoch. Default 0.0.
        valid_size : float, optional
            Fraction of data held out for validation. Default 0.0.
        valid_eval_iter : int, optional
            Validate every this many epochs. Default 1.
        patience : int, optional
            Early-stopping patience in epochs. Default 16.
        verbose : int, optional
            Print progress every this many epochs when > 0. Default 0.
        random_state : int, optional
            Seed for train/validation split. Default None.
        shuffle : bool, optional
            Shuffle data between epochs. Default False.
        **kwargs
            Forwarded to ``torch.optim.Adam``.

        Returns
        -------
        tuple[list[float], list[float]]
            Training losses and validation losses (zeros when
            ``valid_size == 0``).
        """

        # Data preparation
        if dataset is None:
            dataset = QSPRDataset(smiles, target_vals, backend)
        if valid_size > 0.0:
            index_train, index_valid = train_test_split(
                [i for i in range(len(dataset))],
                test_size=valid_size,
                random_state=random_state,
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
        if "lr" in kwargs:
            _lr = kwargs.get("lr")
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
                    [i for i in range(len(dataset))],
                    test_size=valid_size,
                    random_state=random_state,
                )
                dataloader_train = DataLoader(
                    Subset(dataset, index_train), batch_size=batch_size, shuffle=True
                )
                dataloader_valid = DataLoader(
                    Subset(dataset, index_valid),
                    batch_size=len(index_valid),
                    shuffle=True,
                )

            train_loss = 0.0
            self.train()

            for b_idx, batch in enumerate(dataloader_train):
                # BATCH BEGIN
                if not CBO.on_batch_begin(b_idx):
                    break

                optimizer.zero_grad()
                pred = self(batch["desc_vals"])
                target = batch["target_val"]

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
                train_loss += loss.detach().item() * len(batch["target_val"])

                # STEP END
                if not CBO.on_step_end(b_idx):
                    break

            # Determine epoch loss for training, validation data.
            # Run epoch-end callbacks first so Validator evaluates this epoch
            # before we record/print ``valid_loss`` (avoids the unset
            # ``sys.maxsize`` sentinel and a one-epoch lag).
            train_loss /= len(dataloader_train.dataset)
            continue_training = CBO.on_epoch_end(epoch)
            if valid_size > 0.0:
                valid_loss = float(_validator._most_recent_loss)
            else:
                valid_loss = 0.0
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if verbose and epoch % verbose == 0:
                print(
                    "Epoch: {} | Train loss: {} | Valid loss: {}".format(
                        epoch, train_loss, valid_loss
                    )
                )

            if not continue_training:
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
        """
        Compute mean squared error between predicted and target values.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values, shape ``(n_samples, n_features)``.
        target : torch.Tensor
            Target values, shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            MSE loss.
        """

        return F.mse_loss(pred, target)

    def save(self, model_filename: str):
        """
        Saves the model for later use

        Args:
            model_filename (str): filename/path to save model
        """

        if _TORCH_MODEL_FN.match(model_filename) is None:
            raise ValueError("Models must be saved with a `.pt` extension")
        payload = {
            "format": _STATE_FORMAT,
            "arch": {
                "input_dim": self._input_dim,
                "output_dim": self._output_dim,
                "hidden_dim": self._hidden_dim,
                "n_hidden": self._n_hidden,
                "dropout": self._dropout,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(payload, model_filename)


def load_model(model_filename: str) -> ECNet:
    """
    Loads a model for use

    Args:
        model_filename (str): filename/path to load model from

    Notes:
        Accepts legacy full-module ``.pt`` pickles and the preferred
        ``ecnet-state-v1`` state-dict payload written by :meth:`ECNet.save`.
    """

    # weights_only=False: required for legacy full-module pickles (Q8 shim).
    payload = torch.load(model_filename, map_location="cpu", weights_only=False)
    if isinstance(payload, ECNet):
        payload.eval()
        return payload
    if isinstance(payload, dict) and payload.get("format") == _STATE_FORMAT:
        arch = payload["arch"]
        model = ECNet(
            arch["input_dim"],
            arch["output_dim"],
            arch["hidden_dim"],
            arch["n_hidden"],
            dropout=arch.get("dropout", 0.0),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
    raise ValueError(
        "Unrecognized ECNet checkpoint; expected a legacy ECNet pickle or "
        f"an {_STATE_FORMAT!r} state-dict payload"
    )
