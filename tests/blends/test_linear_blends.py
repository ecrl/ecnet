"""Golden tests for linear blend property predictors (CN, YSI, LHV).

Sources cited in ``ecnet.blends.predict`` (linear volume-fraction mixing):

- Cetane number: NREL/SR-540-36805
- Yield sooting index: https://doi.org/10.1016/j.fuel.2020.119522
- Lower heating value: https://doi.org/10.1016/j.ejpe.2015.11.002

Fixtures are self-consistency oracles (hand-computed ``sum(V_i * x_i)``),
not literature table lookups. See ``tests/fixtures/linear_blends.py``.
"""

from __future__ import annotations

import pytest
from tests.fixtures.linear_blends import LINEAR_BLEND_CASES, LinearBlendCase

from ecnet.blends import cetane_number, lower_heating_value, yield_sooting_index

_PREDICTORS = [
    pytest.param(cetane_number, id="cetane_number"),
    pytest.param(yield_sooting_index, id="yield_sooting_index"),
    pytest.param(lower_heating_value, id="lower_heating_value"),
]


@pytest.mark.parametrize("predictor", _PREDICTORS)
@pytest.mark.parametrize(
    "case",
    LINEAR_BLEND_CASES,
    ids=[c.case_id for c in LINEAR_BLEND_CASES],
)
def test_linear_blend_oracle(predictor, case: LinearBlendCase) -> None:
    result = predictor(case.values, case.vol_fractions)
    assert result == pytest.approx(case.expected, abs=case.abs_tol, rel=0.0)
