from pathlib import Path

import pandas as pd

from lenscluster.validation import load_chires_family_summary, load_chires_table


def test_load_chires_table_parses_image_and_summary_rows(tmp_path: Path) -> None:
    path = tmp_path / "chires.dat"
    path.write_text(
        "\n".join(
            [
                "chi multiples",
                " N    ID    z   Narcs    chip    chix    chiy    chia   rmss     rmsi    dx      dy    nwarn",
                " 6    13c 1.005   1     21.67    0.00    0.00    0.00   0.343    0.00    0.12   -0.32  1",
                " 6     13 1.005   3     43.20    0.00    0.00    0.00   0.279    0.00    N/A     N/A   3",
            ]
        ),
        encoding="utf-8",
    )

    table = load_chires_table(path)
    summary = load_chires_family_summary(path)

    assert table.shape[0] == 2
    assert summary.shape[0] == 1
    assert summary.loc[0, "family_id"] == "13"
    assert summary.loc[0, "n_arcs"] == 3
    assert summary.loc[0, "source_rms_arcsec"] == 0.279
    assert pd.isna(summary.loc[0, "dx_arcsec"])
