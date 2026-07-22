"""Self-consistency fixtures for blend error-propagation helpers.

Oracle class: **self-consistency** — hand evaluation of the formulas in
``ecnet.blends.equations`` (not literature table lookups). Expected values are
computed independently of ``ecnet``; ``abs=1e-12`` covers float ``sqrt`` noise.
"""

from __future__ import annotations

from math import sqrt
from typing import NamedTuple


class LinearErrCase(NamedTuple):
    case_id: str
    errors: list[float]
    proportions: list[float]
    expected: float
    abs_tol: float
    note: str


class ExponentialErrCase(NamedTuple):
    case_id: str
    values: list[float]
    result: float
    errors: list[float]
    proportions: list[float]
    a: float
    b: float
    expected: float
    abs_tol: float
    note: str


class KvErrCase(NamedTuple):
    case_id: str
    values: list[float]
    errors: list[float]
    proportions: list[float]
    expected: float
    abs_tol: float
    note: str


_ABS = 1e-12

# linear_blend_err: sqrt(sum((e_i * V_i)^2))
LINEAR_ERR_CASES: list[LinearErrCase] = [
    LinearErrCase(
        case_id="linear_single",
        errors=[2.0],
        proportions=[1.0],
        expected=2.0,  # sqrt((2*1)^2)
        abs_tol=_ABS,
        note="Single-component identity",
    ),
    LinearErrCase(
        case_id="linear_binary",
        errors=[1.0, 2.0],
        proportions=[0.5, 0.5],
        # sqrt((0.5)^2 + (1.0)^2) = sqrt(0.25 + 1.0) = sqrt(1.25)
        expected=sqrt(1.25),
        abs_tol=_ABS,
        note="Equal binary proportions",
    ),
]

# exponential_blend_err: sqrt(sum(((f*b*e_i/x_i)*V_i)^2)); a is unused in body
EXPONENTIAL_ERR_CASES: list[ExponentialErrCase] = [
    ExponentialErrCase(
        case_id="exp_single",
        values=[10.0],
        result=8.0,
        errors=[0.5],
        proportions=[1.0],
        a=1.0,
        b=2.0,
        # |(8*2*0.5/10)*1| = 0.8
        expected=0.8,
        abs_tol=_ABS,
        note="Single-component; a unused by implementation",
    ),
    ExponentialErrCase(
        case_id="exp_binary",
        values=[10.0, 20.0],
        result=12.0,
        errors=[1.0, 2.0],
        proportions=[0.25, 0.75],
        a=1.0,
        b=13.45,
        # term0: (12*13.45*1/10)*0.25 = 4.035
        # term1: (12*13.45*2/20)*0.75 = 12.105
        # sqrt(4.035^2 + 12.105^2)
        expected=sqrt(4.035**2 + 12.105**2),
        abs_tol=_ABS,
        note="Binary unequal; b chosen like CP exponent for numeric variety",
    ),
]

# kv_error: sqrt(sum((V_i * e_i / x_i)^2))
KV_ERR_CASES: list[KvErrCase] = [
    KvErrCase(
        case_id="kv_err_single",
        values=[2.0],
        errors=[0.4],
        proportions=[1.0],
        expected=0.2,  # 1*0.4/2
        abs_tol=_ABS,
        note="Single-component identity",
    ),
    KvErrCase(
        case_id="kv_err_binary",
        values=[2.0, 4.0],
        errors=[0.2, 0.8],
        proportions=[0.5, 0.5],
        # terms: 0.5*0.2/2 = 0.05; 0.5*0.8/4 = 0.1; sqrt(0.05^2 + 0.1^2)
        expected=sqrt(0.05**2 + 0.1**2),
        abs_tol=_ABS,
        note="Equal binary proportions",
    ),
]
