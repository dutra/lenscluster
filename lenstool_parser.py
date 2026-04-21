"""Compatibility shim for legacy imports.

The implementation lives in :mod:`lenscluster.lenstool_parser`.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lenscluster.lenstool_parser import *  # noqa: F401,F403
