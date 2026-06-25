from __future__ import annotations

from importlib import resources

import matplotlib.pyplot as plt


def apply_lenscluster_plot_style() -> None:
    style_path = resources.files("lenscluster").joinpath("style.mplstyle")
    with resources.as_file(style_path) as path:
        plt.style.use(str(path))
