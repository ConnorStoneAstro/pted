from .pted import pted, pted_coverage_test
from .tests import test

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0-dev"

__author__ = "Connor Stone"
__email__ = "connorstone628@gmail.com"

__all__ = [
    "pted",
    "pted_coverage_test",
    "test",
]
