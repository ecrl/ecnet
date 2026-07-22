# Test fixtures

Numeric and file fixtures for ECNet characterization tests live here.

## Oracle labeling

Label every fixture (in the fixture module docstring, filename, or adjacent
comment) with exactly one of the following classes:

| Class | Meaning | Typical use |
|-------|---------|-------------|
| **Self-consistency** | Expected values from an independent reimplementation of the same equations already in `ecnet`, or from hand evaluation of those formulas on fixed inputs | Blend algebraic checks; error-propagation helpers |
| **Literature** | Values taken from a cited paper, table, or dataset card (with DOI or bibliographic key) | Cross-checks against published blend or property tables |

Do not mix classes in one fixture without stating which entries are which.
Regression anchors that only freeze current library output (no independent
derivation) should be labeled **self-consistency / regression** and must not be
presented as literature measurements.

## Tolerances

Record `rel` / `abs` for `pytest.approx` (or equivalent) next to each expected
value, with a short justification (algebraic exactness, float round-trip, or
reported experimental uncertainty from the source).

## Layout

Prefer small Python modules or data files under this directory, imported by
tests under `tests/blends/`, `tests/datasets/`, and related packages. Keep
PaDEL-backed integration inputs minimal; default CI assumes the PaDEL path only.
