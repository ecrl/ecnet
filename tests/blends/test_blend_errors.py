"""Golden tests for blend error-propagation helpers.

Formulas follow ``ecnet.blends.equations`` docstrings. Fixtures are
self-consistency oracles (hand-computed); see ``tests/fixtures/blend_errors.py``.
"""

from __future__ import annotations

import pytest
from tests.fixtures.blend_errors import (
    EXPONENTIAL_ERR_CASES,
    KV_ERR_CASES,
    LINEAR_ERR_CASES,
    ExponentialErrCase,
    KvErrCase,
    LinearErrCase,
)

from ecnet.blends import exponential_blend_err, kv_error, linear_blend_err


@pytest.mark.parametrize(
    "case",
    LINEAR_ERR_CASES,
    ids=[c.case_id for c in LINEAR_ERR_CASES],
)
def test_linear_blend_err_oracle(case: LinearErrCase) -> None:
    result = linear_blend_err(case.errors, case.proportions)
    assert result == pytest.approx(case.expected, abs=case.abs_tol, rel=0.0)


@pytest.mark.parametrize(
    "case",
    EXPONENTIAL_ERR_CASES,
    ids=[c.case_id for c in EXPONENTIAL_ERR_CASES],
)
def test_exponential_blend_err_oracle(case: ExponentialErrCase) -> None:
    result = exponential_blend_err(
        case.values,
        case.result,
        case.errors,
        case.proportions,
        case.a,
        case.b,
    )
    assert result == pytest.approx(case.expected, abs=case.abs_tol, rel=0.0)


@pytest.mark.parametrize(
    "case",
    KV_ERR_CASES,
    ids=[c.case_id for c in KV_ERR_CASES],
)
def test_kv_error_oracle(case: KvErrCase) -> None:
    result = kv_error(case.values, case.errors, case.proportions)
    assert result == pytest.approx(case.expected, abs=case.abs_tol, rel=0.0)
