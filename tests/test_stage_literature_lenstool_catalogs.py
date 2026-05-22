from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "stage_literature_lenstool_catalogs.py"
spec = importlib.util.spec_from_file_location("stage_literature_lenstool_catalogs", SCRIPT_PATH)
assert spec is not None
stager = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = stager
spec.loader.exec_module(stager)


def test_stage_literature_catalogs_copies_only_needed_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_dir = source_root / "Example" / "A2744"
    source_dir.mkdir(parents=True)
    (source_dir / "obs_arcs.cat").write_text("#REFERENCE 0\n1.1 1 2 0.1 0.1 0 2.0 25\n", encoding="utf-8")
    (source_dir / "img_mul_constraints.cat").write_text("#REFERENCE 0\n2.1 1 2 0.1 0.1 0 2.0 25\n", encoding="utf-8")
    (source_dir / "CM_A2744.cat").write_text("#REFERENCE 0\n1 1 2 1 1 0 20 1\n", encoding="utf-8")
    (source_dir / "best.par").write_text("runmode\n reference 3 1 2\n end\n", encoding="utf-8")
    (source_dir / "README.txt").write_text("provenance\n", encoding="utf-8")
    (source_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (source_dir / "bayes.dat").write_text("chain\n", encoding="utf-8")
    (source_dir / "map.fits").write_bytes(b"fits")

    manifest = stager.stage_literature_catalogs(
        source_root,
        tmp_path / "out",
        sources=[
            stager.LiteratureSource(
                "a2744",
                "example",
                "Example/A2744",
                ("*",),
            )
        ],
    )

    copied = manifest.loc[manifest["status"] == "copied", "copied_path"].map(Path)
    copied_names = {path.name for path in copied}
    assert copied_names == {"obs_arcs.cat", "img_mul_constraints.cat", "CM_A2744.cat", "best.par", "README.txt"}
    roles = manifest.set_index("copied_path")["file_role"].to_dict()
    img_path = tmp_path / "out" / "a2744" / "example" / "img_mul_constraints.cat"
    assert roles[str(img_path)] == "image_catalog"
    assert "index.html" not in copied_names
    assert "bayes.dat" not in copied_names
    assert "map.fits" not in copied_names
    assert (tmp_path / "out" / "a2744" / "example" / "obs_arcs.cat").exists()
    assert (tmp_path / "out" / stager.MANIFEST_NAME).exists()


def test_stage_literature_catalogs_reports_missing_m0717(tmp_path: Path) -> None:
    manifest = stager.stage_literature_catalogs(
        tmp_path / "source",
        tmp_path / "out",
        sources=[],
    )

    row = manifest.set_index("cluster").loc["m0717"]
    assert row["status"] == "missing_source"
    assert row["source_slug"] == "no_local_literature_catalog"
