from pathlib import Path

from lenscluster.plotting import plot_path


def test_plot_path_creates_directory(tmp_path: Path) -> None:
    output = plot_path(tmp_path / "plots", "summary.png")

    assert output == tmp_path / "plots" / "summary.png"
    assert output.parent.is_dir()
