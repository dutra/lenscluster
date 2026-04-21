"""Compatibility CLI shim for the packaged solver.

The implementation lives in :mod:`lenscluster.cluster_solver`.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lenscluster.cluster_solver import main


if __name__ == "__main__":
    main()
