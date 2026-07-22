"""Golden tests for nonlinear blend predictors (cloud point, kinematic viscosity).

Sources cited in ``ecnet.blends.predict``:

- Cloud point (°C in / °C out; Rankine internally): Semwal et al., diesel
  blending cold-flow model with exponent 13.45
- Kinematic viscosity (cSt): Ding et al., equation 8 mixing rule

Fixtures are self-consistency oracles (independent stdlib reimplementation),
not literature table lookups. See ``tests/fixtures/nonlinear_blends.py``.
"""

from __future__ import annotations

import pytest
from tests.fixtures.nonlinear_blends import (
    CLOUD_POINT_CASES,
    KINEMATIC_VISCOSITY_CASES,
    NonlinearBlendCase,
)

from ecnet.blends import cloud_point, kinematic_viscosity


@pytest.mark.parametrize(
    "case",
    CLOUD_POINT_CASES,
    ids=[c.case_id for c in CLOUD_POINT_CASES],
)
def test_cloud_point_oracle(case: NonlinearBlendCase) -> None:
    result = cloud_point(case.values, case.vol_fractions)
    assert result == pytest.approx(case.expected, rel=case.rel_tol, abs=case.abs_tol)


@pytest.mark.parametrize(
    "case",
    KINEMATIC_VISCOSITY_CASES,
    ids=[c.case_id for c in KINEMATIC_VISCOSITY_CASES],
)
def test_kinematic_viscosity_oracle(case: NonlinearBlendCase) -> None:
    result = kinematic_viscosity(case.values, case.vol_fractions)
    assert result == pytest.approx(case.expected, rel=case.rel_tol, abs=case.abs_tol)
