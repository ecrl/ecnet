"""Signature locks for the frozen public API (design §8.1–§8.2).

Compares ``inspect.signature`` parameter names, kinds, and defaults.
Annotations are intentionally not locked (e.g. ``List[str]`` vs ``list[str]``).
"""

from __future__ import annotations

import inspect
from inspect import Parameter

import pytest

from ecnet import ECNet
from ecnet.blends import (
    cetane_number,
    cloud_point,
    exponential_blend_err,
    kinematic_viscosity,
    kv_error,
    linear_blend_err,
    lower_heating_value,
    yield_sooting_index,
)
from ecnet.callbacks import Callback, CallbackOperator, LRDecayLinear, Validator
from ecnet.datasets import (
    QSPRDataset,
    QSPRDatasetFromFile,
    QSPRDatasetFromValues,
    load_bp,
    load_cn,
    load_cp,
    load_kv,
    load_lhv,
    load_mon,
    load_mp,
    load_pp,
    load_ron,
    load_ysi,
)
from ecnet.model import load_model
from ecnet.tasks import (
    select_rfr,
    tune_batch_size,
    tune_model_architecture,
    tune_training_parameters,
)

# (name, kind, default) — use inspect.Parameter.empty for required params
_EMPTY = Parameter.empty
_POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
_VAR_KEYWORD = Parameter.VAR_KEYWORD


def _assert_signature(fn, expected: list[tuple]) -> None:
    """Assert parameter names, kinds, and defaults match ``expected``."""
    sig = inspect.signature(fn)
    actual = [
        (name, param.kind, param.default) for name, param in sig.parameters.items()
    ]
    assert actual == expected, (
        f"Signature drift for {getattr(fn, '__qualname__', fn)}:\n"
        f"  expected={expected}\n"
        f"  actual  ={actual}"
    )


def _pok(name: str, default=_EMPTY) -> tuple:
    return (name, _POSITIONAL_OR_KEYWORD, default)


def _varkw(name: str = "kwargs") -> tuple:
    return (name, _VAR_KEYWORD, _EMPTY)


_LOADERS = [
    load_bp,
    load_cn,
    load_cp,
    load_kv,
    load_lhv,
    load_mon,
    load_mp,
    load_pp,
    load_ron,
    load_ysi,
]

_BLEND_PREDICTORS = [
    cetane_number,
    cloud_point,
    kinematic_viscosity,
    lower_heating_value,
    yield_sooting_index,
]


def test_ecnet_init_signature() -> None:
    _assert_signature(
        ECNet.__init__,
        [
            _pok("self"),
            _pok("input_dim"),
            _pok("output_dim"),
            _pok("hidden_dim"),
            _pok("n_hidden"),
            _pok("dropout", 0.0),
            _pok("device", "cpu"),
        ],
    )


def test_ecnet_fit_signature() -> None:
    _assert_signature(
        ECNet.fit,
        [
            _pok("self"),
            _pok("smiles", None),
            _pok("target_vals", None),
            _pok("dataset", None),
            _pok("backend", "padel"),
            _pok("batch_size", 32),
            _pok("epochs", 100),
            _pok("lr_decay", 0.0),
            _pok("valid_size", 0.0),
            _pok("valid_eval_iter", 1),
            _pok("patience", 16),
            _pok("verbose", 0),
            _pok("random_state", None),
            _pok("shuffle", False),
            _varkw(),
        ],
    )


def test_ecnet_forward_signature() -> None:
    _assert_signature(
        ECNet.forward,
        [
            _pok("self"),
            _pok("x"),
        ],
    )


def test_ecnet_save_signature() -> None:
    _assert_signature(
        ECNet.save,
        [
            _pok("self"),
            _pok("model_filename"),
        ],
    )


def test_load_model_signature() -> None:
    _assert_signature(
        load_model,
        [
            _pok("model_filename"),
        ],
    )


@pytest.mark.parametrize("loader", _LOADERS, ids=[fn.__name__ for fn in _LOADERS])
def test_load_prop_signature(loader) -> None:
    _assert_signature(
        loader,
        [
            _pok("as_dataset", False),
            _pok("backend", "padel"),
        ],
    )


def test_qsprdataset_init_signature() -> None:
    _assert_signature(
        QSPRDataset.__init__,
        [
            _pok("self"),
            _pok("smiles"),
            _pok("target_vals"),
            _pok("backend", "padel"),
        ],
    )


def test_qsprdataset_from_file_init_signature() -> None:
    _assert_signature(
        QSPRDatasetFromFile.__init__,
        [
            _pok("self"),
            _pok("smiles_fn"),
            _pok("target_vals"),
            _pok("backend", "padel"),
        ],
    )


def test_qsprdataset_from_values_init_signature() -> None:
    _assert_signature(
        QSPRDatasetFromValues.__init__,
        [
            _pok("self"),
            _pok("desc_vals"),
            _pok("target_vals"),
        ],
    )


def test_select_rfr_signature() -> None:
    _assert_signature(
        select_rfr,
        [
            _pok("dataset"),
            _pok("total_importance", 0.95),
            _varkw(),
        ],
    )


@pytest.mark.parametrize(
    "tune_fn",
    [tune_batch_size, tune_model_architecture, tune_training_parameters],
    ids=["tune_batch_size", "tune_model_architecture", "tune_training_parameters"],
)
def test_tune_helper_signatures(tune_fn) -> None:
    _assert_signature(
        tune_fn,
        [
            _pok("n_bees"),
            _pok("n_iter"),
            _pok("dataset_train"),
            _pok("dataset_eval"),
            _pok("n_processes", 1),
            _varkw(),
        ],
    )


@pytest.mark.parametrize(
    "blend_fn",
    _BLEND_PREDICTORS,
    ids=[fn.__name__ for fn in _BLEND_PREDICTORS],
)
def test_blend_predictor_signatures(blend_fn) -> None:
    _assert_signature(
        blend_fn,
        [
            _pok("values"),
            _pok("vol_fractions"),
        ],
    )


def test_linear_blend_err_signature() -> None:
    _assert_signature(
        linear_blend_err,
        [
            _pok("errors"),
            _pok("proportions"),
        ],
    )


def test_exponential_blend_err_signature() -> None:
    _assert_signature(
        exponential_blend_err,
        [
            _pok("values"),
            _pok("result"),
            _pok("errors"),
            _pok("proportions"),
            _pok("a"),
            _pok("b"),
        ],
    )


def test_kv_error_signature() -> None:
    _assert_signature(
        kv_error,
        [
            _pok("values"),
            _pok("errors"),
            _pok("proportions"),
        ],
    )


def test_callback_init_signature() -> None:
    _assert_signature(Callback.__init__, [_pok("self")])


def test_callback_operator_init_signature() -> None:
    _assert_signature(CallbackOperator.__init__, [_pok("self")])


def test_lr_decay_linear_init_signature() -> None:
    _assert_signature(
        LRDecayLinear.__init__,
        [
            _pok("self"),
            _pok("init_lr"),
            _pok("decay_rate"),
            _pok("optimizer"),
        ],
    )


def test_validator_init_signature() -> None:
    _assert_signature(
        Validator.__init__,
        [
            _pok("self"),
            _pok("loader"),
            _pok("model"),
            _pok("eval_iter"),
            _pok("patience"),
        ],
    )


def test_ecnet_getattr_unknown_attribute() -> None:
    import ecnet

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = ecnet.not_a_public_symbol


def test_public_surface_importable() -> None:
    """Sanity: every §8.1 symbol remains importable under its public path."""
    from ecnet import __version__
    from ecnet.blends import (  # noqa: F401
        cetane_number,
        cloud_point,
        exponential_blend_err,
        kinematic_viscosity,
        kv_error,
        linear_blend_err,
        lower_heating_value,
        yield_sooting_index,
    )
    from ecnet.callbacks import (  # noqa: F401
        Callback,
        CallbackOperator,
        LRDecayLinear,
        Validator,
    )
    from ecnet.datasets import (  # noqa: F401 — re-check package exports
        QSPRDataset,
        QSPRDatasetFromFile,
        QSPRDatasetFromValues,
        load_bp,
        load_cn,
        load_cp,
        load_kv,
        load_lhv,
        load_mon,
        load_mp,
        load_pp,
        load_ron,
        load_ysi,
    )
    from ecnet.tasks import (  # noqa: F401
        select_rfr,
        tune_batch_size,
        tune_model_architecture,
        tune_training_parameters,
    )

    assert isinstance(__version__, str)
    assert __version__
    # PCADataset must remain an advanced import, not a datasets package export.
    import ecnet.datasets as datasets_pkg

    assert not hasattr(datasets_pkg, "PCADataset")
