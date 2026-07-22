"""Sphinx configuration for ECNet."""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Third-party import noise must not fail `sphinx-build -W`.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*TorchScript.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*reduce_op.*",
    category=FutureWarning,
)

project = "ECNet"
author = "Travis Kessler"
copyright = f"{datetime.now(tz=timezone.utc).year}, {author}"
release = "4.1.5"
version = "4.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path: list[str] = ["_static"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Avoid pulling optional descriptor stacks into every autodoc import graph.
autodoc_mock_imports = [
    "padelpy",
    "alvadescpy",
    "ecabc",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]


def _sanitize_autodoc_docstrings(app, what, name, obj, options, lines):
    """Normalize legacy Google-style docstrings for Sphinx warning-as-error builds."""
    cleaned: list[str] = []
    for line in lines:
        text = line.replace("**kwargs", "kwargs")
        text = text.replace("(*, 1)", "(N, 1)")
        if "$$" in text:
            text = text.replace("$$", "")
        cleaned.append(text)
    # Keep summary + Args/Returns, but drop bracketed kwargs catalogs that break RST.
    out: list[str] = []
    skipping_catalog = False
    for line in cleaned:
        stripped = line.strip()
        if "kwargs can include any in" in stripped:
            skipping_catalog = True
            out.append("Additional keyword arguments are forwarded to training.")
            continue
        if skipping_catalog:
            if stripped.startswith("Args:") or stripped.startswith("Returns:"):
                skipping_catalog = False
            else:
                continue
        out.append(line)
    lines[:] = out


def setup(app):
    app.connect("autodoc-process-docstring", _sanitize_autodoc_docstrings)
