# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.1.5] — 2026-07-22

First modernization compatibility release. Public import signatures remain
stable relative to `4.1.4`; packaging, tests, docs, and the release path are
brought up to the approved Design A–E baseline.

### Added

- Contract / characterization test suite (API signatures, blends oracles,
  datasets, model save/load, callbacks, tasks) with CI coverage fail-under 90%.
- `[dev]` and `[docs]` extras; Ruff and pre-commit configuration.
- Sphinx + Furo documentation (`docs/source/`), Read the Docs config, dataset
  cards, units notes, and API stability page.
- Governance files: `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CITATION.cff`, issue/PR templates.
- Dependabot; CI `pip-audit` with documented torch exceptions
  (`docs/SECURITY_EXCEPTIONS.md`).
- PyPI trusted publishing via OIDC (`.github/workflows/release.yml`); maintainer
  steps in `docs/RELEASING.md`.
- Refreshed `examples/` notebooks (seeded demos, train-only scaling where
  needed, stripped outputs).

### Changed

- Package layout moved to `src/ecnet/`; `ecnet.__version__` via
  `importlib.metadata`.
- Runtime dependencies use compatible version ranges instead of exact pins
  (`torch>=2.4.0,<2.6`, `scikit-learn>=1.5.1,<2`, and matching ranges for
  `padelpy`, `alvadescpy`, and `ecabc`).
- `ECNet.save` writes an `ecnet-state-v1` checkpoint (architecture metadata plus
  `state_dict`). `load_model` still loads legacy full-module `.pt` pickles with
  the same signature and prediction behavior.
- CI matrix on Ubuntu for Python 3.11–3.12 (lint, tests, audit, docs).
- Descriptor backends (`padel` / `alvadesc`) are imported lazily so PaDEL-only
  installs are not blocked by `alvadescpy` / `pkg_resources`.
- `databases/` documented as a research archive (not installed in wheels).

### Fixed

- Validation loss is recorded after each epoch’s `Validator` evaluation (no
  `sys.maxsize` sentinel / one-epoch lag in returned histories).
- `QSPRDataset.set_index` / `set_desc_index` use tensor indexing (avoids slow
  list-of-ndarray tensor construction warnings).
- Training and save/load tests no longer leave fixed-name temporary files in the
  process working directory.

[Unreleased]: https://github.com/ecrl/ecnet/compare/4.1.5...HEAD
[4.1.5]: https://github.com/ecrl/ecnet/compare/4.1.4...4.1.5
