r"""Training callback objects/functions"""
import sys


class CallbackOperator(object):
    """
    CallbackOperator: executes individual callback steps at each step
    """

    def __init__(self):

        self.cb = []

    def add_cb(self, cb):

        self.cb.append(cb)

    def on_train_begin(self):

        for cb in self.cb:
            if not cb.on_train_begin():
                return False
        return True

    def on_train_end(self):

        for cb in self.cb:
            if not cb.on_train_end():
                return False
        return True

    def on_epoch_begin(self, epoch):

        for cb in self.cb:
            if not cb.on_epoch_begin(epoch):
                return False
        return True

    def on_epoch_end(self, epoch):

        for cb in self.cb:
            if not cb.on_epoch_end(epoch):
                return False
        return True

    def on_batch_begin(self, batch):

        for cb in self.cb:
            if not cb.on_batch_begin(batch):
                return False
        return True

    def on_batch_end(self, batch):

        for cb in self.cb:
            if not cb.on_batch_end(batch):
                return False
        return True

    def on_loss_begin(self, batch):

        for cb in self.cb:
            if not cb.on_loss_begin(batch):
                return False
        return True

    def on_loss_end(self, batch):

        for cb in self.cb:
            if not cb.on_loss_end(batch):
                return False
        return True

    def on_step_begin(self, batch):

        for cb in self.cb:
            if not cb.on_step_begin(batch):
                return False
        return True

    def on_step_end(self, batch):

        for cb in self.cb:
            if not cb.on_step_end(batch):
                return False
        return True


class Callback(object):
    """
    Base Callback object
    """

    def __init__(self): pass
    def on_train_begin(self): return True
    def on_train_end(self): return True
    def on_epoch_begin(self, epoch): return True
    def on_epoch_end(self, epoch): return True
    def on_batch_begin(self, batch): return True
    def on_batch_end(self, batch): return True
    def on_loss_begin(self, batch): return True
    def on_loss_end(self, batch): return True
    def on_step_begin(self, batch): return True
    def on_step_end(self, batch): return True


class LRDecayLinear(Callback):

    def __init__(self, init_lr: float, decay_rate: float, optimizer):
        """
        Linear learning rate decay

        Args:
            init_lr (float): initial learning rate
            decay_rate (float): decay per epoch
            optimizer (torch.optim.Adam): optimizer used for training
        """
        super().__init__()
        self._init_lr = init_lr
        self._decay = decay_rate
        self.optimizer = optimizer

    def on_epoch_begin(self, epoch: int) -> bool:
        """
        Training halted if:
            new learing rate == 0.0
        """

        lr = max(0.0, self._init_lr - epoch * self._decay)
        if lr == 0.0:
            return False
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return True


class Validator(Callback):

    def __init__(self, loader, model, eval_iter: int, patience: int):
        """
        Periodic validation using training data subset

        Args:
            loader (torch.utils.data.DataLoader): validation set
            model (ecnet.ECNet): model being trained
            eval_iter (int): validation set evaluated after `this` many epochs
            patience (int): if new lowest validation loss not found after `this` many epochs,
                terminate training, set model parameters to those observed @ lowest validation loss
        """

        super().__init__()
        self.loader = loader
        self.model = model
        self._ei = eval_iter
        self._best_loss = sys.maxsize
        self._most_recent_loss = sys.maxsize
        self._epoch_since_best = 0
        self.best_state = model.state_dict()
        self._patience = patience

    def on_epoch_end(self, epoch: int) -> bool:
        """
        Training halted if:
            number of epochs since last lowest valid. MAE > specified patience
        """

        if epoch % self._ei != 0:
            return True
        valid_loss = 0.0
        for batch in self.loader:
            v_pred = self.model(batch['desc_vals'])
            v_target = batch['target_val']
            v_loss = self.model.loss(v_pred, v_target)
            valid_loss += v_loss * len(batch['target_val'])
        valid_loss /= len(self.loader.dataset)
        self._most_recent_loss = valid_loss
        if valid_loss < self._best_loss:
            self._best_loss = valid_loss
            self.best_state = self.model.state_dict()
            self._epoch_since_best = 0
            return True
        self._epoch_since_best += self._ei
        if self._epoch_since_best > self._patience:
            return False
        return True

    def on_train_end(self) -> bool:
        """
        After training, recall weights when lowest valid. MAE occurred
        """

        self.model.load_state_dict(self.best_state)
        return True
