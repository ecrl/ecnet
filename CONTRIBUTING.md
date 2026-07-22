# Contributing to ECNet

Thank you for contributing. This note covers local development, checks, and
pull-request expectations for the current compatibility series.

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md). Security issues
should be reported privately per [SECURITY.md](SECURITY.md).

## Development environment

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Optional documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Checks before opening a pull request

```bash
pre-commit run --all-files
ruff check src tests
ruff format --check src tests
pytest tests/ -v --cov=ecnet --cov-report=term-missing
```

Docs (when you change Sphinx pages or docstrings that affect autodoc). The CI
`docs` job runs the same command with warnings as errors:

```bash
sphinx-build -W -b html docs/source docs/_build/html
```

Dependency audit (same pattern as the CI `audit` job; ignores are listed in
`docs/SECURITY_EXCEPTIONS.md`):

```bash
ignore_args=()
while IFS= read -r id; do
  [[ -z "$id" || "$id" =~ ^# ]] && continue
  ignore_args+=(--ignore-vuln "$id")
done < docs/pip-audit-ignores.txt
pip-audit "${ignore_args[@]}"
```

Hooks run Ruff lint/format and basic file hygiene on staged changes. Configure
them once with `pre-commit install` after installing the `[dev]` extra.

## Example notebooks

Notebooks under `examples/` demonstrate the current v4 API with the default
PaDEL backend (Java required). They are **not** run in CI.

Manual smoke (optional, after `pip install -e ".[dev]"` and a working Java
install):

```bash
jupyter execute examples/getting_started.ipynb
# or open the notebooks in Jupyter and Run All
```

Commit notebooks **without** stored outputs. Do not reintroduce
`backend="alvadesc"` in committed examples unless the change is clearly marked
as license-gated.

## Pull request checklist

- [ ] Tests added or updated for new behavior
- [ ] Docs updated if user-facing behavior changed
- [ ] `CHANGELOG.md` `[Unreleased]` entry for user-visible changes
- [ ] Public import signatures remain stable within the current compatibility
      series unless a design note documents otherwise
- [ ] Example notebook outputs left cleared when notebooks change

File issues for bugs or feature requests using the GitHub templates. Include
OS, Python version, and relevant error output for bugs.

## Releasing

Maintainers: see [docs/RELEASING.md](docs/RELEASING.md) for PyPI trusted
publishing (OIDC), the `pypi` GitHub Environment, and cutting a GitHub Release.
