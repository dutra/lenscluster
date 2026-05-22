from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_hff_to_literature.py"
spec = importlib.util.spec_from_file_location("compare_hff_to_literature", SCRIPT_PATH)
assert spec is not None
compare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = compare
spec.loader.exec_module(compare)


def test_default_catalog_root_reads_generated_results() -> None:
    assert compare.DEFAULT_CATALOG_ROOT == Path("results") / "hff_master_catalogs"


def test_parse_lenstool_reference0_image_catalog(tmp_path: Path) -> None:
    path = tmp_path / "obs_arcs.cat"
    path.write_text(
        "#REFERENCE 0\n"
        "1.1 10.0 -1.0 0.1 0.1 0.0 2.0 25.0\n"
        "1.2 10.1 -1.1 0.1 0.1 0.0 2.0 25.0\n",
        encoding="utf-8",
    )

    catalog = compare.parse_lenstool_catalog(path, catalog_kind="image")

    assert catalog["literature_id"].tolist() == ["1.1", "1.2"]
    assert catalog["family_id"].tolist() == ["1", "1"]
    np.testing.assert_allclose(catalog["ra"], [10.0, 10.1])
    np.testing.assert_allclose(catalog["catalog_z"], [2.0, 2.0])
    assert catalog["catalog_quality"].tolist() == ["", ""]


def test_parse_plain_image_catalog_with_id_ra_dec_z(tmp_path: Path) -> None:
    path = tmp_path / "sl-final.dat"
    path.write_text("# ID RA DEC z cat\n1.1 10.0 -1.0 2.5 Gold\n", encoding="utf-8")

    catalog = compare.parse_lenstool_catalog(path, catalog_kind="image")

    assert len(catalog) == 1
    assert catalog.iloc[0]["literature_id"] == "1.1"
    assert catalog.iloc[0]["family_id"] == "1"
    np.testing.assert_allclose(catalog.iloc[0]["catalog_z"], 2.5)
    assert np.isnan(catalog.iloc[0]["catalog_mag"])
    assert catalog.iloc[0]["catalog_quality"] == "Gold"


def test_parse_lenstool_reference3_member_catalog(tmp_path: Path) -> None:
    path = tmp_path / "members.cat"
    path.write_text("#REFERENCE 3 10.0 20.0\n1 0.0 3600.0 1.0 1.0 0.0 21.0 1.0\n", encoding="utf-8")

    catalog = compare.parse_lenstool_catalog(path, catalog_kind="member")

    assert catalog.iloc[0]["literature_id"] == "1"
    np.testing.assert_allclose(catalog.iloc[0]["ra"], 10.0)
    np.testing.assert_allclose(catalog.iloc[0]["dec"], 21.0)
    np.testing.assert_allclose(catalog.iloc[0]["catalog_mag"], 21.0)


def test_parse_plain_schuldt_member_table(tmp_path: Path) -> None:
    path = tmp_path / "Cluster_members_final.txt"
    path.write_text(
        "# header\n"
        "ID R.A. Decl. F160W eF160W Sel.\n"
        "--------------------------------------------------\n"
        "1 177.3731103 22.3929993 21.17 0.01 P\n",
        encoding="utf-8",
    )

    catalog = compare.parse_plain_member_table(path)

    assert len(catalog) == 1
    assert catalog.iloc[0]["literature_id"] == "1"
    np.testing.assert_allclose(catalog.iloc[0]["ra"], 177.3731103)
    assert catalog.iloc[0]["selection"] == "P"


def test_sky_match_nearest_resolves_duplicate_by_smallest_separation() -> None:
    left = pd.DataFrame({"ra": [10.0, 10.0 + 0.20 / 3600.0], "dec": [0.0, 0.0]})
    right = pd.DataFrame({"ra": [10.0 + 0.05 / 3600.0], "dec": [0.0]})

    matches = compare.sky_match_nearest(left, right, radius_arcsec=1.0)

    assert len(matches) == 1
    assert matches.iloc[0]["left_index"] == 0
    assert matches.iloc[0]["right_index"] == 0
    np.testing.assert_allclose(matches.iloc[0]["separation_arcsec"], 0.05, atol=1.0e-6)


def test_family_overlap_classifies_agreed_partial_and_unmatched() -> None:
    image_matches = pd.DataFrame(
        [
            {"cluster": "a370", "source_id": "src", "source_slug": "s", "match_type": "our", "our_family_id": "F1", "literature_family_id": "1", "matched": True},
            {"cluster": "a370", "source_id": "src", "source_slug": "s", "match_type": "our", "our_family_id": "F1", "literature_family_id": "1", "matched": True},
            {"cluster": "a370", "source_id": "src", "source_slug": "s", "match_type": "our", "our_family_id": "F1", "literature_family_id": "1", "matched": True},
            {"cluster": "a370", "source_id": "src", "source_slug": "s", "match_type": "our", "our_family_id": "F1", "literature_family_id": "2", "matched": True},
            {"cluster": "a370", "source_id": "src", "source_slug": "s", "match_type": "our", "our_family_id": "F2", "literature_family_id": "3", "matched": True},
            {"cluster": "a370", "source_id": "src", "source_slug": "s", "match_type": "our", "our_family_id": "F2", "literature_family_id": "4", "matched": True},
            {"cluster": "a370", "source_id": "src", "source_slug": "s", "match_type": "our", "our_family_id": "F3", "literature_family_id": "", "matched": False},
        ]
    )

    overlap = compare.family_overlap_from_image_matches(image_matches).set_index("our_family_id")

    assert overlap.loc["F1", "agreement"] == "agreed"
    assert overlap.loc["F2", "agreement"] == "partial"
    assert overlap.loc["F3", "agreement"] == "unmatched"


def _write_mini_catalogs(catalog_root: Path, literature_root: Path) -> None:
    our_dir = catalog_root / "a370"
    our_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"object_id": "m1", "ra": 10.0, "dec": 0.0, "member_for_lensing": True},
            {"object_id": "m2", "ra": 10.1, "dec": 0.0, "member_for_lensing": False},
        ]
    ).to_csv(our_dir / "a370_cluster_members.csv", index=False)
    pd.DataFrame(
        [
            {"object_id": "i1", "candidate_family_id": "F1", "ra": 20.0, "dec": 0.0},
            {"object_id": "i2", "candidate_family_id": "F1", "ra": 20.1, "dec": 0.0},
        ]
    ).to_csv(our_dir / "a370_candidate_family_members.csv", index=False)

    lit_dir = literature_root / "a370" / "example"
    lit_dir.mkdir(parents=True)
    member_path = lit_dir / "members.cat"
    image_path = lit_dir / "mul.cat"
    bad_path = lit_dir / "bad.cat"
    member_path.write_text("#REFERENCE 0\nL1 10.0001 0.0 1 1 0 20 1\n", encoding="utf-8")
    image_path.write_text(
        "#REFERENCE 0\n"
        "1.1 20.0001 0.0 0.1 0.1 0 2.0 25.0\n"
        "1.2 20.1001 0.0 0.1 0.1 0 2.0 25.0\n",
        encoding="utf-8",
    )
    bad_path.write_text("#REFERENCE 9\n1 1 2 3 4 5 6 7\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"cluster": "a370", "source_slug": "example", "file_role": "member_catalog", "source_path": "", "copied_path": member_path, "size_bytes": 1, "copied_at": "", "status": "copied"},
            {"cluster": "a370", "source_slug": "example", "file_role": "image_catalog", "source_path": "", "copied_path": image_path, "size_bytes": 1, "copied_at": "", "status": "copied"},
            {"cluster": "a370", "source_slug": "example", "file_role": "image_catalog", "source_path": "", "copied_path": bad_path, "size_bytes": 1, "copied_at": "", "status": "copied"},
            {"cluster": "m0717", "source_slug": "no_local_literature_catalog", "file_role": "missing_source", "source_path": "", "copied_path": "", "size_bytes": 0, "copied_at": "", "status": "missing_source"},
        ]
    ).to_csv(literature_root / compare.MANIFEST_NAME, index=False)


def test_compare_all_writes_matches_summaries_and_plots(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalogs"
    literature_root = tmp_path / "literature"
    out_dir = tmp_path / "plots"
    _write_mini_catalogs(catalog_root, literature_root)

    sources, member_matches, image_matches, family_overlap, cluster_summary = compare.compare_all(
        catalog_root,
        literature_root,
        clusters=("a370", "m0717"),
    )
    _sources_again, catalogs = compare.discover_literature_catalogs(literature_root)
    compare.write_outputs(out_dir, sources, member_matches, image_matches, family_overlap, cluster_summary, catalog_root, catalogs)

    assert set(sources["status"]) >= {"parsed", "parse_error", "missing_source"}
    assert member_matches["matched"].map(compare._bool_value).sum() == 2
    assert set(member_matches.loc[member_matches["matched"].map(compare._bool_value), "subset"]) == {"members", "lensing"}
    assert image_matches["matched"].map(compare._bool_value).sum() == 2
    assert family_overlap.iloc[0]["agreement"] == "agreed"
    assert "missing_literature_source" in set(cluster_summary["status"])
    assert (out_dir / "literature_sources.csv").exists()
    assert (out_dir / "member_matches.csv").exists()
    assert (out_dir / "image_matches.csv").exists()
    assert (out_dir / "family_overlap.csv").exists()
    assert (out_dir / "cluster_summary.csv").exists()
    assert any(out_dir.rglob("*.png"))
