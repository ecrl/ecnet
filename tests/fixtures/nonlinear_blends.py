"""Self-consistency fixtures for cloud point and kinematic viscosity blends.

Oracle class: **self-consistency** — independent stdlib reimplementation of the
equations documented in ``ecnet.blends.predict`` (not literature table lookups,
and not imported from ``ecnet.blends.equations``).

Units
-----
- Cloud point I/O: °C (internal Rankine conversion in the oracle mirrors Semwal)
- Kinematic viscosity I/O: cSt (Ding et al. mixing rule, equation 8)

Tolerances use ``rel=1e-12`` and ``abs=1e-12`` for float ``**`` / ``log`` / ``exp``.
"""

from __future__ import annotations

from math import exp, log
from typing import NamedTuple


class NonlinearBlendCase(NamedTuple):
    """One nonlinear blend oracle case."""

    case_id: str
    values: list[float]
    vol_fractions: list[float]
    expected: float
    rel_tol: float
    abs_tol: float
    note: str


_REL = 1e-12
_ABS = 1e-12


def _celsius_to_rankine(temp_c: float) -> float:
    return (9 / 5) * temp_c + 491.67


def _rankine_to_celsius(temp_r: float) -> float:
    return (temp_r - 491.67) * (1 / (9 / 5))


def _oracle_cloud_point_c(values_c: list[float], vol_fractions: list[float]) -> float:
    """Semwal-style CP blend; inputs/outputs in °C, power sum in Rankine."""
    cp_sum = 0.0
    for idx, val in enumerate(values_c):
        cp_sum += vol_fractions[idx] * _celsius_to_rankine(val) ** 13.45
    return _rankine_to_celsius(cp_sum ** (1 / 13.45))


def _oracle_kinematic_viscosity_cst(
    values_cst: list[float], vol_fractions: list[float]
) -> float:
    """Ding et al. equation 8; kinematic viscosity in cSt."""
    kv_sum = 0.0
    for idx, val in enumerate(values_cst):
        kv_sum += vol_fractions[idx] / log(2000 * val)
    return exp(1 / kv_sum) / 2000


def _case(
    case_id: str,
    values: list[float],
    vol_fractions: list[float],
    expected: float,
    note: str,
) -> NonlinearBlendCase:
    return NonlinearBlendCase(
        case_id=case_id,
        values=values,
        vol_fractions=vol_fractions,
        expected=expected,
        rel_tol=_REL,
        abs_tol=_ABS,
        note=note,
    )


# Cloud point (°C): Semwal et al. diesel blending model (see predict.py docstring).
_CP_SINGLE_VALUES = [5.0]
_CP_SINGLE_VOLS = [1.0]
_CP_BINARY_VALUES = [-10.0, 20.0]
_CP_BINARY_VOLS = [0.3, 0.7]

CLOUD_POINT_CASES: list[NonlinearBlendCase] = [
    _case(
        "cp_single_component_celsius",
        _CP_SINGLE_VALUES,
        _CP_SINGLE_VOLS,
        _oracle_cloud_point_c(_CP_SINGLE_VALUES, _CP_SINGLE_VOLS),
        "Identity in °C; exercises Rankine round-trip consistency",
    ),
    _case(
        "cp_binary_unequal_celsius",
        _CP_BINARY_VALUES,
        _CP_BINARY_VOLS,
        _oracle_cloud_point_c(_CP_BINARY_VALUES, _CP_BINARY_VOLS),
        "Binary CP blend; values and result in °C",
    ),
]

# Kinematic viscosity (cSt): Ding et al. equation 8.
_KV_SINGLE_VALUES = [2.5]
_KV_SINGLE_VOLS = [1.0]
_KV_BINARY_VALUES = [1.5, 4.0]
_KV_BINARY_VOLS = [0.4, 0.6]

KINEMATIC_VISCOSITY_CASES: list[NonlinearBlendCase] = [
    _case(
        "kv_single_component_cst",
        _KV_SINGLE_VALUES,
        _KV_SINGLE_VOLS,
        _oracle_kinematic_viscosity_cst(_KV_SINGLE_VALUES, _KV_SINGLE_VOLS),
        "Identity in cSt",
    ),
    _case(
        "kv_binary_unequal_cst",
        _KV_BINARY_VALUES,
        _KV_BINARY_VOLS,
        _oracle_kinematic_viscosity_cst(_KV_BINARY_VALUES, _KV_BINARY_VOLS),
        "Binary KV blend; values and result in cSt",
    ),
]
