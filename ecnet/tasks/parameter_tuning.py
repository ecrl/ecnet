from ecabc import ABC
from sklearn.metrics import median_absolute_error
from copy import deepcopy

from ..model import ECNet
from ..datasets.structs import QSPRDataset

from typing import Iterable


CONFIG = {
    'training_params_range': {
        'lr': (0.0, 0.05),
        'lr_decay': (0.0, 0.0001)
    },
    'architecture_params_range': {
        'hidden_dim': (1, 1024),
        'n_hidden': (1, 5),
        'dropout': (0.0, 0.1)
    }
}


def _get_kwargs(**kwargs):
    """
    Returns dictionary of relevant training parameters from **kwargs

    Args:
        **kwargs: key word arguments

    Returns:
        dict: relevant relevant kwargs, else default values
    """

    return {
        'model': kwargs.get('model'),
        'train_ds': kwargs.get('train_ds'),
        'eval_ds': kwargs.get('eval_ds'),
        'epochs': kwargs.get('epochs', 100),
        'batch_size': kwargs.get('batch_size', 32),
        'lr_decay': kwargs.get('lr_decay', 0.0),
        'lr': kwargs.get('lr', 0.001),
        'beta_1': kwargs.get('beta_1', 0.9),
        'beta_2': kwargs.get('beta_2', 0.999),
        'eps': kwargs.get('eps', 1e-08),
        'weight_decay': kwargs.get('weight_decay', 0.0),
        'hidden_dim': kwargs.get('hidden_dim', 128),
        'n_hidden': kwargs.get('n_hidden', 2),
        'dropout': kwargs.get('dropout', 0.0),
        'amsgrad': kwargs.get('amsgrad', False)
    }


def _evaluate_model(trial_spec: dict) -> float:
    """
    Training sub-function for cost functions _cost_batch_size, _cost_arch, _cost_train_hp;

    Args:
        trial_spec (dict): all relevant parameters for this training trial

    Returns:
        float: median absolute error for dataset being evaluated (trial_spec['eval_ds'])
    """

    model = deepcopy(trial_spec['model'])
    model._hidden_dim = trial_spec['hidden_dim']
    model._n_hidden = trial_spec['n_hidden']
    model._dropout = trial_spec['dropout']
    model._construct()
    model.fit(
        dataset=trial_spec['train_ds'],
        epochs=trial_spec['epochs'],
        batch_size=trial_spec['batch_size'],
        lr_decay=trial_spec['lr_decay'],
        lr=trial_spec['lr_decay'],
        betas=(trial_spec['beta_1'], trial_spec['beta_2']),
        eps=trial_spec['eps'],
        weight_decay=trial_spec['weight_decay'],
        amsgrad=trial_spec['amsgrad']
    )
    yhat_eval = model(trial_spec['eval_ds'].desc_vals).detach().numpy()
    y_eval = trial_spec['eval_ds'].target_vals
    return median_absolute_error(y_eval, yhat_eval)


def _cost_batch_size(vals: Iterable[float], **kwargs) -> float:
    """
    Cost function for tuning batch size

    Args:
        vals (iterable[float]): values passed to cost function from ABC; just contains batch size
        **kwargs: user-defined training arguments, datasets to be passed to _evaluate_model

    Returns:
        float: median absolute error for dataset being evaluated (**kwarg: eval_ds)
    """

    trial_spec = _get_kwargs(**kwargs)
    trial_spec['batch_size'] = vals[0]
    return _evaluate_model(trial_spec)


def tune_batch_size(n_bees: int, n_iter: int, n_processes: int = 1, **kwargs) -> dict:
    """
    Tunes the batch size during training

    Args:
        n_bees (int): number of employer bees to use in ABC algorithm
        n_iter (int): number of iterations, or "search cycles", for ABC algorithm
        n_processes (int): if > 1, uses multiprocessing when evaluating at an iteration
        **kwargs: arguments passed to _cost_batch_size

    Returns:
        dict: {'batch_size': tuned batch size}
    """

    abc = ABC(n_bees, _cost_batch_size, num_processes=n_processes, obj_fn_args=kwargs)
    abc.add_param(1, len(kwargs.get('train_ds').desc_vals), name='batch_size')
    abc.initialize()
    for _ in range(n_iter):
        abc.search()
    return {'batch_size': abc.best_params['batch_size']}


def _cost_arch(vals, **kwargs):
    """
    Cost function for tuning NN architecture

    Args:
        vals (iterable[float]): values passed to cost function from ABC; contains:
            - hidden_dim
            - n_nidden
            - dropout
        **kwargs: user-defined training arguments, datasets to be passed to _evaluate_model

    Returns:
        float: median absolute error for dataset being evaluated (**kwarg: eval_ds)
    """

    trial_spec = _get_kwargs(**kwargs)
    trial_spec['hidden_dim'] = vals[0]
    trial_spec['n_hidden'] = vals[1]
    trial_spec['dropout'] = vals[2]
    return _evaluate_model(trial_spec)


def tune_model_architecture(n_bees: int, n_iter: int, n_processes: int = 1, **kwargs) -> dict:
    """
    Tunes the NN's architecture

    Args:
        n_bees (int): number of employer bees to use in ABC algorithm
        n_iter (int): number of iterations, or "search cycles", for ABC algorithm
        n_processes (int): if > 1, uses multiprocessing when evaluating at an iteration
        **kwargs: arguments passed to _cost_batch_size

    Returns:
        dict: {'batch_size': opt_val, 'n_hidden': opt_val, 'dropout': opt_val}
    """

    abc = ABC(n_bees, _cost_arch, num_processes=n_processes, obj_fn_args=kwargs)
    abc.add_param(CONFIG['architecture_params_range']['hidden_dim'][0],
                  CONFIG['architecture_params_range']['hidden_dim'][1], name='hidden_dim')
    abc.add_param(CONFIG['architecture_params_range']['n_hidden'][0],
                  CONFIG['architecture_params_range']['n_hidden'][1], name='n_hidden')
    abc.add_param(CONFIG['architecture_params_range']['dropout'][0],
                  CONFIG['architecture_params_range']['dropout'][1], name='dropout')
    abc.initialize()
    for _ in range(n_iter):
        abc.search()
    return {
        'hidden_dim': abc.best_params['hidden_dim'],
        'n_hidden': abc.best_params['n_hidden'],
        'dropout': abc.best_params['dropout']
    }


def _cost_train_hp(vals, **kwargs):
    """
    Cost function for tuning NN training parameters (Adam optim. hyper-parameters)

    Args:
        vals (iterable[float]): values passed to cost function from ABC; contains:
            - lr (learning rate)
            - lr_decay (learning rate decay)
        **kwargs: user-defined training arguments, datasets to be passed to _evaluate_model

    Returns:
        float: median absolute error for dataset being evaluated (**kwarg: eval_ds)
    """

    trial_spec = _get_kwargs(**kwargs)
    trial_spec['lr'] = vals[0]
    trial_spec['lr_decay'] = vals[1]
    return _evaluate_model(trial_spec)


def tune_training_parameters(n_bees: int, n_iter: int, n_processes: int = 1, **kwargs) -> dict:
    """
    Tunes the NN's training parameters (Adam optim. fn.)

    Args:
        n_bees (int): number of employer bees to use in ABC algorithm
        n_iter (int): number of iterations, or "search cycles", for ABC algorithm
        n_processes (int): if > 1, uses multiprocessing when evaluating at an iteration
        **kwargs: arguments passed to _cost_batch_size

    Returns:
        dict: {'lr': opt_val, 'lr_decay': opt_val}
    """

    abc = ABC(n_bees, _cost_train_hp, num_processes=n_processes, obj_fn_args=kwargs)
    abc.add_param(CONFIG['training_params_range']['lr'][0],
                  CONFIG['training_params_range']['lr'][1], name='lr')
    abc.add_param(CONFIG['training_params_range']['lr_decay'][0],
                  CONFIG['training_params_range']['lr_decay'][1], name='lr_decay')
    abc.initialize()
    for _ in range(n_iter):
        abc.search()
    return {
        'lr': abc.best_params['lr'],
        'lr_decay': abc.best_params['lr_decay']
    }
