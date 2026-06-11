from pathlib import Path

import pytest

import download_catalogs as downloader


def test_build_pagul2024_targets_filters_cluster_by_short_key(tmp_path: Path) -> None:
    targets = downloader.build_pagul2024_targets(tmp_path, cluster="a370")

    assert len(targets) == 2
    assert {target.cluster for target in targets} == {"abell370"}
    assert {target.filename for target in targets} == {
        "hlsp_buffalo_hst_ir-weighted_abell370_multi_v2.0_catalog.fits",
        "hlsp_buffalo_hst_ir-weighted_abell370_multi_v2.0_readme.txt",
    }
    assert {target.destination.parent for target in targets} == {tmp_path}


def test_build_hff_redshift_targets_filters_cluster_and_service(tmp_path: Path) -> None:
    targets = downloader.build_hff_redshift_targets(tmp_path, services="ned", cluster="a370")

    assert len(targets) == 2
    assert {target.cluster.key for target in targets} == {"a370"}
    assert {target.cluster.target for target in targets} == {"abell370"}
    assert {target.field for target in targets} == {"core", "parallel"}
    assert {target.service for target in targets} == {"ned"}
    assert {target.destination.parent for target in targets} == {tmp_path / "a370"}


def test_build_buffalo_image_targets_filters_cluster_and_fetches_one_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched_urls: list[str] = []
    fetched_timeouts: list[float] = []

    def fake_fetch_text(url: str, *, timeout: float) -> str:
        fetched_urls.append(url)
        fetched_timeouts.append(timeout)
        return (
            "curl --output abell370/hlsp_buffalo_hst_f814w_abell370_drz.fits "
            "https://example.invalid/hlsp_buffalo_hst_f814w_abell370_drz.fits\n"
        )

    monkeypatch.setattr(downloader, "fetch_text", fake_fetch_text)

    targets = downloader.build_buffalo_image_targets(
        tmp_path,
        image_scale="30mas",
        cluster="a370",
        timeout=12.0,
    )

    assert fetched_urls == [downloader.buffalo_image_script_url("abell370", "30mas")]
    assert fetched_timeouts == [12.0]
    assert len(targets) == 1
    assert targets[0].cluster == "abell370"
    assert targets[0].destination == tmp_path / "abell370" / "hlsp_buffalo_hst_f814w_abell370_drz.fits"


def test_parse_args_accepts_short_and_mast_cluster_names() -> None:
    assert downloader.parse_args(["--cluster", "a370"]).cluster == "a370"
    assert downloader.parse_args(["--cluster", "abell370"]).cluster == "abell370"


def test_parse_args_rejects_unknown_cluster() -> None:
    with pytest.raises(SystemExit):
        downloader.parse_args(["--cluster", "not-a-cluster"])
