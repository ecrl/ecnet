from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ECNet as ECNet

__version__ = version("ecnet")

__all__ = ["ECNet", "__version__"]


def __getattr__(name: str):
    if name == "ECNet":
        from .model import ECNet as _ECNet

        return _ECNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
