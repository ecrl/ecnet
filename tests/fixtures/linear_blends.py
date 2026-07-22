"""Self-consistency fixtures for linear volume-fraction blend averages.

Oracle class: **self-consistency** (hand evaluation of ``sum(V_i * x_i)``).
These are not literature table values.

The public predictors ``cetane_number``, ``yield_sooting_index``, and
``lower_heating_value`` all use this linear mixing rule (see ``ecnet.blends.predict``
docstrings for NREL / DOI citations). Expected values below are computed
independently of ``ecnet``; tolerances assume algebraic float summation.
"""

from __future__ import annotations

from typing import NamedTuple


class LinearBlendCase(NamedTuple):
    """One linear blend oracle case."""

    case_id: str
    values: list[float]
    vol_fractions: list[float]
    expected: float
    abs_tol: float
    note: str


# abs=1e-12: exact weighted sums for these inputs within double float noise.
_ABS = 1e-12

LINEAR_BLEND_CASES: list[LinearBlendCase] = [
    LinearBlendCase(
        case_id="binary_equal",
        values=[40.0, 60.0],
        vol_fractions=[0.5, 0.5],
        expected=50.0,  # 0.5*40 + 0.5*60
        abs_tol=_ABS,
        note="Equal binary blend",
    ),
    LinearBlendCase(
        case_id="ternary_unequal",
        values=[30.0, 50.0, 70.0],
        vol_fractions=[0.5, 0.3, 0.2],
        expected=44.0,  # 0.5*30 + 0.3*50 + 0.2*70 = 15 + 15 + 14
        abs_tol=_ABS,
        note="Unequal ternary blend; fractions sum to 1.0",
    ),
    LinearBlendCase(
        case_id="single_component",
        values=[55.0],
        vol_fractions=[1.0],
        expected=55.0,
        abs_tol=_ABS,
        note="Identity: single component with volume fraction 1.0",
    ),
]
