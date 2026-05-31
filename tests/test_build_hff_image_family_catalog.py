from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_hff_master_catalog.py"
spec = importlib.util.spec_from_file_location("build_hff_master_catalog_for_families", SCRIPT_PATH)
assert spec is not None
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


TEST_BANDS = ["F435W", "F475W", "F606W", "F625W", "F814W", "F105W", "F125W", "F160W"]


def _row(
    object_id: str,
    *,
    offset_arcsec: float,
    zspec: float = np.nan,
    rank: float = np.nan,
    confidence: str = "",
    zphot: float = np.nan,
    zphot_low: float = np.nan,
    zphot_high: float = np.nan,
    mags: list[float] | None = None,
    member_zspec: bool = False,
    member_photo: bool = False,
) -> dict[str, object]:
    base_mags = mags if mags is not None else [24.0, 24.2, 24.3, 24.2, 24.1, 23.9, 23.8, 23.7]
    row: dict[str, object] = {
        "cluster_key": "a370",
        "object_id": object_id,
        "ra": 10.0 + offset_arcsec / 3600.0,
        "dec": 0.0,
        "zspec_best": zspec,
        "zspec_best_confidence_rank": rank,
        "zspec_best_confidence": confidence,
        "zspec_best_native_quality": rank,
        "zspec_best_source": "pagul2024" if np.isfinite(rank) and rank >= 1.0 else "shipley2018",
        "zspec_selection_note": "selected_by_normalized_confidence",
        "zphot_best": zphot,
        "pagul_zpdf": zphot,
        "pagul_zpdf_low": zphot_low,
        "pagul_zpdf_high": zphot_high,
        "pagul_nb_used": 6.0 if np.isfinite(zphot) else np.nan,
        "member_zspec_candidate": member_zspec,
        "member_photoz_candidate": member_photo,
        "object_source": "pagul2024",
        "catalog_sources": "pagul2024",
    }
    for band, mag in zip(TEST_BANDS, base_mags):
        row[f"mag_{band}"] = mag
    return row


def _assert_pair_frames_close(left: pd.DataFrame, right: pd.DataFrame) -> None:
    left = left.sort_values(["left_object_id", "right_object_id"]).reset_index(drop=True)
    right = right.sort_values(["left_object_id", "right_object_id"]).reset_index(drop=True)
    assert left[["cluster_key", "left_object_id", "right_object_id", "hard_reject_reason", "redshift_relation"]].equals(
        right[["cluster_key", "left_object_id", "right_object_id", "hard_reject_reason", "redshift_relation"]]
    )
    numeric_columns = [
        "separation_arcsec",
        "separation_kpc",
        "pair_score",
        "specz_score",
        "photoz_score",
        "zphot_delta",
        "color_score",
        "sed_rms",
        "n_common_bands",
    ]
    for column in numeric_columns:
        np.testing.assert_allclose(left[column], right[column], rtol=1.0e-10, atol=1.0e-10, equal_nan=True)


def _score_pairs(catalog: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    spec = builder.CLUSTER_BY_KEY["a370"]
    candidates = builder.prepare_candidates(catalog, spec)
    return builder.score_candidate_pairs(
        candidates,
        spec,
        family_pair_diagnostics="all",
        **kwargs,
    )


def _score_two_rows(left: dict[str, object], right: dict[str, object], **kwargs: object) -> pd.Series:
    pairs = _score_pairs(pd.DataFrame([left, right]), **kwargs)
    assert len(pairs) == 1
    return pairs.iloc[0]


def _score_raw_pair(left: dict[str, object], right: dict[str, object]) -> pd.Series:
    pairs = builder.score_candidate_pairs(
        pd.DataFrame([left, right]),
        builder.CLUSTER_BY_KEY["a370"],
        family_pair_diagnostics="all",
        pair_score_threshold=0.0,
    )
    assert len(pairs) == 1
    return pairs.iloc[0]


def _accepted_pair(
    left: str,
    right: str,
    score: float,
    *,
    specz_score: float = 1.0,
    color_score: float = 1.0,
    sed_rms: float = 0.02,
    separation_arcsec: float = 10.0,
) -> dict[str, object]:
    return {
        "cluster_key": "a370",
        "left_object_id": left,
        "right_object_id": right,
        "separation_arcsec": separation_arcsec,
        "separation_kpc": separation_arcsec * builder.kpc_per_arcsec(builder.CLUSTER_BY_KEY["a370"].z_lens),
        "pair_score": score,
        "specz_score": specz_score,
        "photoz_score": 0.0,
        "zphot_delta": np.nan,
        "color_score": color_score,
        "sed_rms": sed_rms,
        "n_common_bands": len(TEST_BANDS),
        "hard_reject_reason": "",
        "redshift_relation": "both_specz",
    }


def _canonical_family_sets(families: list[set[str]]) -> list[tuple[str, ...]]:
    return sorted(tuple(sorted(family)) for family in families)


def _reference_complete_link_family(
    seed: tuple[str, str],
    adjacency: dict[str, dict[str, float]],
) -> set[str]:
    family = set(seed)
    candidate_pool = set(adjacency[seed[0]]) | set(adjacency[seed[1]])
    candidate_pool.difference_update(family)
    while candidate_pool:
        compatible = [
            candidate
            for candidate in candidate_pool
            if all(member in adjacency[candidate] for member in family)
        ]
        if not compatible:
            break
        compatible.sort(
            key=lambda candidate: (
                -float(np.mean([adjacency[candidate][member] for member in family])),
                candidate,
            )
        )
        chosen = compatible[0]
        family.add(chosen)
        candidate_pool = {
            candidate
            for candidate in candidate_pool
            if candidate != chosen and all(member in adjacency[candidate] for member in family)
        }
    return family


def _reference_growth_sets(
    pairs: pd.DataFrame,
    *,
    two_image_score_threshold: float = 0.0,
) -> list[set[str]]:
    sorted_pairs = pairs.sort_values("pair_score", ascending=False, kind="mergesort").reset_index(drop=True)
    adjacency: dict[str, dict[str, float]] = {}
    for row in sorted_pairs.itertuples(index=False):
        left = str(row.left_object_id)
        right = str(row.right_object_id)
        adjacency.setdefault(left, {})[right] = float(row.pair_score)
        adjacency.setdefault(right, {})[left] = float(row.pair_score)
    families: dict[frozenset[str], set[str]] = {}
    for row in sorted_pairs.itertuples(index=False):
        family = _reference_complete_link_family((str(row.left_object_id), str(row.right_object_id)), adjacency)
        if len(family) == 2 and float(row.pair_score) < two_image_score_threshold:
            continue
        if len(family) == 2 and (
            float(row.specz_score) < 0.70 or float(row.sed_rms) > builder.FAMILY_COLOR_RMS_ACCEPTABLE
        ):
            continue
        families[frozenset(family)] = family
    return list(families.values())


def test_jax_dense_growth_kernel_returns_packed_masks_and_sizes() -> None:
    score_matrix = np.array(
        [
            [0.0, 0.97, 0.91],
            [0.97, 0.0, 0.86],
            [0.91, 0.86, 0.0],
        ],
        dtype=np.float32,
    )
    adjacency_matrix = score_matrix > 0.0
    packed, sizes = builder.jax.device_get(
        builder._grow_complete_link_family_batch_jax(
            builder.jnp.asarray(score_matrix),
            builder.jnp.asarray(adjacency_matrix),
            builder.jnp.asarray(np.array([0, 0], dtype=np.int32)),
            builder.jnp.asarray(np.array([1, 0], dtype=np.int32)),
            builder.jnp.asarray(np.array([True, False])),
        )
    )
    decoded = np.unpackbits(np.asarray(packed), axis=1, count=3).astype(bool)

    assert np.asarray(packed).shape == (2, 1)
    assert np.asarray(sizes).tolist() == [3, 0]
    assert decoded[0].tolist() == [True, True, True]
    assert decoded[1].tolist() == [False, False, False]


def test_photoz_only_pair_is_scored_with_perfect_colors() -> None:
    secure_pair = _score_two_rows(
        _row("s1", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row("s2", offset_arcsec=10.0, zspec=2.001, rank=3, confidence="secure"),
    )
    photo_pair = _score_raw_pair(
        {**_row("p1", offset_arcsec=0.0, zspec=np.nan, rank=np.nan, zphot=2.0, zphot_low=1.8, zphot_high=2.2), "image_zphot_family": 2.0},
        {**_row("p2", offset_arcsec=10.0, zspec=np.nan, rank=np.nan, zphot=2.01, zphot_low=1.8, zphot_high=2.2), "image_zphot_family": 2.01},
    )

    assert secure_pair["hard_reject_reason"] == ""
    assert photo_pair["hard_reject_reason"] == ""
    assert photo_pair["redshift_relation"] == "photoz_only"
    assert photo_pair["pair_score"] >= builder.DEFAULT_PAIR_SCORE_THRESHOLD
    assert secure_pair["specz_score"] > 0.8
    assert photo_pair["specz_score"] == 0.0
    assert photo_pair["photoz_score"] > 0.0


def test_secure_or_probable_specz_conflict_hard_rejects_pair() -> None:
    pair = _score_two_rows(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row("b", offset_arcsec=10.0, zspec=2.02, rank=3, confidence="secure"),
    )

    assert pair["hard_reject_reason"] == "secure_or_probable_specz_conflict"
    assert pair["pair_score"] == 0.0


def test_specz_between_excellent_and_hard_tolerance_passes_downgraded() -> None:
    excellent_pair = _score_two_rows(
        _row("e1", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row("e2", offset_arcsec=10.0, zspec=2.001, rank=3, confidence="secure"),
    )
    tolerated_pair = _score_two_rows(
        _row("t1", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row("t2", offset_arcsec=10.0, zspec=2.0075, rank=3, confidence="secure"),
    )

    assert tolerated_pair["hard_reject_reason"] == ""
    assert tolerated_pair["redshift_relation"] == "both_specz"
    assert tolerated_pair["specz_score"] < excellent_pair["specz_score"]
    assert tolerated_pair["pair_score"] < excellent_pair["pair_score"]
    assert tolerated_pair["pair_score"] > 0.0


def test_fallback_specz_is_used_by_pair_scoring_with_quality_penalty() -> None:
    secure_pair = _score_two_rows(
        _row("s1", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row("s2", offset_arcsec=10.0, zspec=2.0, rank=3, confidence="secure"),
    )
    fallback_pair = _score_raw_pair(
        _row("f1", offset_arcsec=0.0, zspec=2.0, rank=0.5, confidence="fallback", zphot=2.0),
        _row("f2", offset_arcsec=10.0, zspec=2.0, rank=0.5, confidence="fallback", zphot=2.0),
    )

    assert fallback_pair["hard_reject_reason"] == ""
    assert fallback_pair["redshift_relation"] == "both_specz"
    assert fallback_pair["specz_score"] == secure_pair["specz_score"]
    assert fallback_pair["pair_score"] < secure_pair["pair_score"]
    assert fallback_pair["pair_score"] > 0.0


def test_probable_specz_agreement_passes_strict_pair_scoring() -> None:
    secure_pair = _score_two_rows(
        _row("s1", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row("s2", offset_arcsec=10.0, zspec=2.0, rank=3, confidence="secure"),
    )
    probable_pair = _score_two_rows(
        _row("p1", offset_arcsec=0.0, zspec=2.0, rank=2, confidence="probable"),
        _row("p2", offset_arcsec=10.0, zspec=2.0, rank=2, confidence="probable"),
    )

    assert probable_pair["hard_reject_reason"] == ""
    assert probable_pair["redshift_relation"] == "both_specz"
    assert probable_pair["photoz_score"] == 0.0
    assert probable_pair["pair_score"] >= builder.DEFAULT_PAIR_SCORE_THRESHOLD
    assert probable_pair["pair_score"] < secure_pair["pair_score"]


def test_prepare_candidates_allows_any_specz_or_qualified_photoz_backgrounds() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    secure_background = _row("secure-bg", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure")
    secure_background["pagul_zpdf"] = np.nan
    probable_background = _row("probable-bg", offset_arcsec=10.0, zspec=2.0, rank=2, confidence="probable")
    secure_foreground = _row("secure-fg", offset_arcsec=20.0, zspec=0.1, rank=3, confidence="secure", zphot=2.0)
    no_strong_photo = _row("photo-bg", offset_arcsec=30.0, zphot=2.0)
    tentative_background = _row("tentative-bg", offset_arcsec=40.0, zspec=2.0, rank=1, confidence="tentative")

    candidates = builder.prepare_candidates(
        pd.DataFrame([secure_background, probable_background, secure_foreground, no_strong_photo, tentative_background]),
        spec,
    )

    assert candidates["object_id"].tolist() == ["secure-bg", "probable-bg", "photo-bg", "tentative-bg"]
    assert np.isnan(candidates.set_index("object_id").loc["secure-bg", "image_zphot_family"])
    assert np.isfinite(candidates.set_index("object_id").loc["photo-bg", "image_zphot_family"])
    metrics = candidates.attrs["image_family_selection_metrics"]
    assert metrics["n_rejected_missing_strong_specz"] == 0
    assert metrics["n_image_family_photoz_companion_candidates"] == 1
    assert metrics["n_image_family_strong_specz_candidates"] == 3


def test_prepare_candidates_admits_only_qualified_photoz_companions() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    good = _row("good-photo", offset_arcsec=0.0, zphot=2.0)
    missing_zpdf = _row("missing-zpdf", offset_arcsec=10.0)
    bad_f160w = _row("bad-f160w", offset_arcsec=20.0, zphot=2.0, mags=[24.0, 24.2, 24.3, 24.2, 24.1, 23.9, 23.8, 26.0])
    low_nb = _row("low-nb", offset_arcsec=30.0, zphot=2.0)
    low_nb["pagul_nb_used"] = 4.0

    candidates = builder.prepare_candidates(pd.DataFrame([good, missing_zpdf, bad_f160w, low_nb]), spec)
    prepped = builder.apply_image_photoz_quality(pd.DataFrame([good, missing_zpdf, bad_f160w, low_nb]))
    reasons = prepped.set_index("object_id")["image_photoz_reject_reason"].to_dict()

    assert candidates["object_id"].tolist() == ["good-photo"]
    assert "missing_or_invalid_zpdf" in reasons["missing-zpdf"]
    assert "f160w_outside_figure9_range" in reasons["bad-f160w"]
    assert "low_nb_used" in reasons["low-nb"]


def test_secure_specz_pair_ignores_contradictory_photoz_when_colors_match() -> None:
    pair = _score_two_rows(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure", zphot=0.2),
        _row("b", offset_arcsec=10.0, zspec=2.0, rank=3, confidence="secure", zphot=5.0),
    )

    assert pair["hard_reject_reason"] == ""
    assert pair["redshift_relation"] == "both_specz"
    assert pair["photoz_score"] == 0.0
    assert pair["color_score"] > 0.9


def test_secure_specz_conflict_rejects_even_with_matching_photoz() -> None:
    pair = _score_two_rows(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure", zphot=2.0),
        _row("b", offset_arcsec=10.0, zspec=2.02, rank=3, confidence="secure", zphot=2.0),
    )

    assert pair["hard_reject_reason"] == "secure_or_probable_specz_conflict"
    assert pair["redshift_relation"] == "secure_or_probable_specz_conflict"


def test_single_specz_pair_with_consistent_photoz_is_scored() -> None:
    spec_anchor = _row("spec", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure", zphot=0.2)
    good_photo = _row("photo", offset_arcsec=10.0, zphot=2.1, zphot_low=1.8, zphot_high=2.3)
    good_photo["image_zphot_family"] = 2.1
    missing_photo = _row("missing-photo", offset_arcsec=10.0)
    missing_photo["image_zphot_family"] = np.nan

    good_pair = _score_raw_pair(spec_anchor, {**good_photo, "zspec_best": np.nan, "zspec_best_confidence_rank": np.nan})
    missing_pair = _score_raw_pair(
        {**spec_anchor, "image_zphot_family": np.nan},
        {**missing_photo, "zspec_best": np.nan, "zspec_best_confidence_rank": np.nan},
    )

    assert good_pair["hard_reject_reason"] == ""
    assert good_pair["redshift_relation"] == "specz_photoz"
    assert good_pair["specz_score"] == 0.0
    assert good_pair["photoz_score"] > 0.0
    assert np.isnan(good_pair["zphot_delta"])
    assert missing_pair["hard_reject_reason"] == "missing_strong_specz"


def test_low_quality_specz_photoz_pair_is_scored() -> None:
    low_quality_anchor = _row("spec", offset_arcsec=0.0, zspec=2.0, rank=0.0, confidence="low_quality", zphot=0.2)
    good_photo = _row("photo", offset_arcsec=10.0, zphot=2.1, zphot_low=1.8, zphot_high=2.3)
    good_photo["image_zphot_family"] = 2.1

    pair = _score_raw_pair(
        low_quality_anchor,
        {**good_photo, "zspec_best": np.nan, "zspec_best_confidence_rank": np.nan},
    )

    assert pair["hard_reject_reason"] == ""
    assert pair["redshift_relation"] == "specz_photoz"
    assert pair["specz_score"] == 0.0
    assert pair["photoz_score"] > 0.0
    assert pair["pair_score"] > 0.0


def test_specz_photoz_pair_requires_photoz_delta_when_both_have_qualified_photoz() -> None:
    spec_anchor = _row("spec", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure", zphot=2.0)
    spec_anchor["image_zphot_family"] = 2.0
    good_photo = _row("photo-good", offset_arcsec=10.0, zphot=2.9, zphot_low=1.8, zphot_high=3.0)
    good_photo["image_zphot_family"] = 2.9
    bad_photo = _row("photo-bad", offset_arcsec=10.0, zphot=3.1, zphot_low=1.8, zphot_high=3.2)
    bad_photo["image_zphot_family"] = 3.1

    good_pair = _score_raw_pair(spec_anchor, {**good_photo, "zspec_best": np.nan, "zspec_best_confidence_rank": np.nan})
    bad_pair = _score_raw_pair(spec_anchor, {**bad_photo, "zspec_best": np.nan, "zspec_best_confidence_rank": np.nan})

    assert good_pair["hard_reject_reason"] == ""
    np.testing.assert_allclose(good_pair["zphot_delta"], 0.9)
    assert bad_pair["hard_reject_reason"] == "photoz_delta_too_large"
    np.testing.assert_allclose(bad_pair["zphot_delta"], 1.1)


def test_both_specz_pair_is_not_vetoed_by_photoz_delta() -> None:
    pair = _score_two_rows(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure", zphot=0.2),
        _row("b", offset_arcsec=10.0, zspec=2.001, rank=3, confidence="secure", zphot=5.0),
    )

    assert pair["hard_reject_reason"] == ""
    assert pair["redshift_relation"] == "both_specz"
    assert pair["zphot_delta"] > builder.DEFAULT_FAMILY_PHOTOZ_DELTA_MAX


def test_no_specz_pair_accepts_consistent_qualified_photoz() -> None:
    good_pair = _score_raw_pair(
        {**_row("a", offset_arcsec=0.0, zspec=np.nan, rank=np.nan, zphot=2.0), "image_zphot_family": 2.0},
        {**_row("b", offset_arcsec=10.0, zspec=np.nan, rank=np.nan, zphot=2.1), "image_zphot_family": 2.1},
    )
    inconsistent_pair = _score_raw_pair(
        {**_row("a", offset_arcsec=0.0, zspec=np.nan, rank=np.nan, zphot=2.0), "image_zphot_family": 2.0},
        {**_row("b", offset_arcsec=10.0, zspec=np.nan, rank=np.nan, zphot=3.0), "image_zphot_family": 3.0},
    )
    missing_pair = _score_raw_pair(
        {**_row("a", offset_arcsec=0.0), "image_zphot_family": np.nan},
        {**_row("b", offset_arcsec=10.0, zphot=2.0), "image_zphot_family": 2.0},
    )

    assert good_pair["hard_reject_reason"] == ""
    assert good_pair["redshift_relation"] == "photoz_only"
    assert good_pair["photoz_score"] > 0.0
    assert inconsistent_pair["hard_reject_reason"] == "photoz_delta_too_large"
    assert missing_pair["hard_reject_reason"] == "missing_strong_specz"


def test_low_quality_specz_conflict_hard_rejects_pair() -> None:
    pair = _score_two_rows(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=0.0, confidence="low_quality"),
        _row("b", offset_arcsec=10.0, zspec=2.02, rank=0.0, confidence="low_quality"),
    )

    assert pair["hard_reject_reason"] == "secure_or_probable_specz_conflict"
    assert pair["pair_score"] == 0.0


def test_family_probability_caps_use_relaxed_moderate_values() -> None:
    high_pair_scores = np.asarray([0.96, 0.97, 0.98], dtype=float)
    low_sed = np.asarray([0.1, 0.1, 0.1], dtype=float)
    high_sed = np.asarray([builder.FAMILY_COLOR_RMS_ACCEPTABLE + 0.05, 0.1, 0.1], dtype=float)
    mixed_pair_scores = np.asarray([0.49, 1.0, 1.0], dtype=float)
    complete_specz = np.asarray([2.0, 2.001, 2.002], dtype=float)
    complete_ranks = np.asarray([3.0, 2.0, 0.0], dtype=float)

    incomplete_probability, incomplete_flags = builder._family_probability_from_arrays(
        3,
        high_pair_scores,
        low_sed,
        np.asarray([2.0, np.nan, 2.001], dtype=float),
        np.asarray([3.0, np.nan, 0.0], dtype=float),
        is_two_image=False,
    )
    generic_large_probability, generic_large_flags = builder._family_probability_from_arrays(
        3,
        high_pair_scores,
        high_sed,
        complete_specz,
        complete_ranks,
        is_two_image=False,
    )
    generic_low_probability, generic_low_flags = builder._family_probability_from_arrays(
        3,
        mixed_pair_scores,
        low_sed,
        complete_specz,
        complete_ranks,
        is_two_image=False,
    )
    photoz_probability, photoz_flags = builder._photoz_only_family_probability_from_arrays(high_pair_scores, low_sed)
    photoz_large_probability, photoz_large_flags = builder._photoz_only_family_probability_from_arrays(
        high_pair_scores,
        high_sed,
    )
    photoz_low_probability, photoz_low_flags = builder._photoz_only_family_probability_from_arrays(
        mixed_pair_scores,
        low_sed,
    )
    anchored_probability, anchored_flags = builder._anchored_family_probability(
        high_pair_scores,
        low_sed,
        max_separation_kpc=30.0,
        max_family_span_kpc=250.0,
        flags=["single_specz_anchor", "incomplete_specz"],
    )
    anchored_large_probability, anchored_large_flags = builder._anchored_family_probability(
        high_pair_scores,
        high_sed,
        max_separation_kpc=230.0,
        max_family_span_kpc=250.0,
        flags=["single_specz_anchor", "incomplete_specz"],
    )
    anchored_low_probability, anchored_low_flags = builder._anchored_family_probability(
        mixed_pair_scores,
        low_sed,
        max_separation_kpc=30.0,
        max_family_span_kpc=250.0,
        flags=["single_specz_anchor", "incomplete_specz"],
    )

    np.testing.assert_allclose(incomplete_probability, 0.93)
    assert "incomplete_specz" in incomplete_flags
    np.testing.assert_allclose(generic_large_probability, 0.85)
    assert "large_sed_residual" in generic_large_flags
    assert generic_low_probability > 0.65
    assert "low_min_pair_score" in generic_low_flags
    np.testing.assert_allclose(photoz_probability, 0.72)
    assert "missing_specz" in photoz_flags
    np.testing.assert_allclose(photoz_large_probability, 0.62)
    assert "large_sed_residual" in photoz_large_flags
    np.testing.assert_allclose(photoz_low_probability, 0.55)
    assert "low_min_pair_score" in photoz_low_flags
    np.testing.assert_allclose(anchored_probability, 0.85)
    assert "single_specz_anchor" in anchored_flags
    np.testing.assert_allclose(anchored_large_probability, 0.75)
    assert "large_sed_residual" in anchored_large_flags
    assert "large_family_span" in anchored_large_flags
    np.testing.assert_allclose(anchored_low_probability, 0.70)
    assert "low_min_pair_score" in anchored_low_flags


def test_constant_magnitude_offset_preserves_color_score() -> None:
    pair = _score_two_rows(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row(
            "b",
            offset_arcsec=10.0,
            zspec=2.0,
            rank=3,
            confidence="secure",
            mags=[value + 1.7 for value in [24.0, 24.2, 24.3, 24.2, 24.1, 23.9, 23.8, 23.7]],
        ),
    )

    assert pair["hard_reject_reason"] == ""
    assert pair["n_common_bands"] == len(TEST_BANDS)
    np.testing.assert_allclose(pair["sed_rms"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(pair["color_score"], 1.0, atol=1.0e-12)


def test_color_score_requires_minimum_common_bands() -> None:
    too_sparse_pair = _score_raw_pair(
        _row(
            "a",
            offset_arcsec=0.0,
            zspec=2.0,
            rank=3,
            confidence="secure",
            mags=[np.nan, 24.2, 24.3, 24.2, 24.1, 23.9, 23.8, 23.7],
        ),
        _row(
            "b",
            offset_arcsec=10.0,
            zspec=2.0,
            rank=3,
            confidence="secure",
            mags=[24.0, np.nan, 24.3, 24.2, 24.1, 23.9, 23.8, 23.7],
        ),
    )
    enough_pair = _score_two_rows(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row(
            "c",
            offset_arcsec=10.0,
            zspec=2.0,
            rank=3,
            confidence="secure",
            mags=[value + 1.7 for value in [24.0, 24.2, 24.3, 24.2, 24.1, 23.9, 23.8, 23.7]],
        ),
    )

    assert too_sparse_pair["pair_score"] == 0.0
    assert too_sparse_pair["n_common_bands"] == 6
    assert too_sparse_pair["hard_reject_reason"] == "insufficient_common_bands"
    assert enough_pair["hard_reject_reason"] == ""
    assert enough_pair["n_common_bands"] == builder.DEFAULT_MIN_COMMON_BANDS
    assert enough_pair["color_score"] > 0.99
    np.testing.assert_allclose(enough_pair["sed_rms"], 0.0, atol=1.0e-12)


def test_catastrophic_color_residual_hard_rejects_pair() -> None:
    base_mags = [24.0, 24.2, 24.3, 24.2, 24.1, 23.9, 23.8, 23.7]
    soft_mags = list(base_mags)
    hard_mags = list(base_mags)
    soft_mags[3] = base_mags[3] - 1.24 * np.sqrt(len(TEST_BANDS))
    hard_mags[3] = base_mags[3] - 1.26 * np.sqrt(len(TEST_BANDS))
    soft_pair = _score_raw_pair(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row(
            "b",
            offset_arcsec=10.0,
            zspec=2.0,
            rank=3,
            confidence="secure",
            mags=soft_mags,
        ),
    )
    hard_pair = _score_raw_pair(
        _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
        _row(
            "c",
            offset_arcsec=10.0,
            zspec=2.0,
            rank=3,
            confidence="secure",
            mags=hard_mags,
        ),
    )

    assert soft_pair["hard_reject_reason"] == ""
    np.testing.assert_allclose(soft_pair["sed_rms"], 1.24, atol=1.0e-6)
    assert soft_pair["color_score"] < 0.5
    assert hard_pair["hard_reject_reason"] == "color_rms_too_large"
    np.testing.assert_allclose(hard_pair["sed_rms"], 1.26, atol=1.0e-6)
    assert hard_pair["pair_score"] == 0.0


def test_jax_pair_scoring_chunking_and_padding_do_not_emit_fake_rows() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("b", offset_arcsec=10.0, zspec=2.001, rank=3, confidence="secure"),
            _row("c", offset_arcsec=20.0, zspec=1.999, rank=3, confidence="secure"),
        ]
    )
    candidates = builder.prepare_candidates(catalog, spec)

    tiny_batch = builder.score_candidate_pairs(
        candidates,
        spec,
        family_pair_diagnostics="all",
        family_pair_batch_size=1,
    )
    padded_batch = builder.score_candidate_pairs(
        candidates,
        spec,
        family_pair_diagnostics="all",
        family_pair_batch_size=8,
    )

    _assert_pair_frames_close(tiny_batch, padded_batch)
    assert len(padded_batch) == 3
    assert padded_batch.attrs["pair_score_metrics"]["pair_score_backend"] == "jax"


def test_pair_scoring_exposes_accepted_arrays_matching_dataframe() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("b", offset_arcsec=10.0, zspec=2.001, rank=3, confidence="secure"),
            _row("conflict", offset_arcsec=20.0, zspec=2.02, rank=3, confidence="secure"),
        ]
    )
    candidates = builder.prepare_candidates(catalog, spec)

    pairs = builder.score_candidate_pairs(
        candidates,
        spec,
        family_pair_diagnostics="scored",
        family_pair_batch_size=2,
    )
    accepted = pairs.loc[(pairs["hard_reject_reason"] == "") & (pairs["pair_score"] >= builder.DEFAULT_PAIR_SCORE_THRESHOLD)]
    arrays = pairs.attrs["accepted_pair_arrays"]
    object_ids = arrays["object_ids"]
    array_pairs = {
        tuple(sorted((str(object_ids[left]), str(object_ids[right]))))
        for left, right in zip(arrays["left_idx"], arrays["right_idx"])
    }
    frame_pairs = {
        tuple(sorted((str(row.left_object_id), str(row.right_object_id))))
        for row in accepted.itertuples(index=False)
    }

    assert array_pairs == frame_pairs
    assert pairs.attrs["pair_score_metrics"]["n_accepted_array_pairs"] == len(accepted)
    assert pairs.attrs["pair_score_metrics"]["n_prefilter_full_score_pairs"] <= pairs.attrs["pair_score_metrics"]["n_spatial_pairs"]


def test_jax_dense_growth_matches_reference_for_two_image_family() -> None:
    pairs = pd.DataFrame([_accepted_pair("a", "b", 0.95)])

    actual, metrics = builder._grow_complete_link_families_jax(pairs, two_image_score_threshold=0.90)
    expected = _reference_growth_sets(pairs, two_image_score_threshold=0.90)

    assert _canonical_family_sets(actual) == _canonical_family_sets(expected)
    assert metrics["family_growth_backend"] == "jax_dense"
    assert metrics["n_family_growth_objects"] == 2
    assert metrics["n_family_growth_seed_edges"] == 1
    assert metrics["family_growth_n_batches"] == 1
    assert metrics["family_growth_unique_masks"] == 1
    assert metrics["family_growth_packed_mask_bytes"] > 0
    assert metrics["family_growth_all_seed_batch"] is True


def test_jax_dense_growth_matches_reference_for_three_image_complete_link_family() -> None:
    pairs = pd.DataFrame(
        [
            _accepted_pair("a", "b", 0.97),
            _accepted_pair("a", "c", 0.91),
            _accepted_pair("b", "c", 0.86),
        ]
    )

    actual, _metrics = builder._grow_complete_link_families_jax(pairs, two_image_score_threshold=0.90)
    expected = _reference_growth_sets(pairs, two_image_score_threshold=0.90)

    assert _canonical_family_sets(actual) == _canonical_family_sets(expected)
    assert _canonical_family_sets(actual) == [("a", "b", "c")]


def test_jax_dense_growth_excludes_incompatible_third_image_like_reference() -> None:
    pairs = pd.DataFrame(
        [
            _accepted_pair("a", "b", 0.97),
            _accepted_pair("b", "c", 0.93),
        ]
    )

    actual, _metrics = builder._grow_complete_link_families_jax(pairs, two_image_score_threshold=0.90)
    expected = _reference_growth_sets(pairs, two_image_score_threshold=0.90)

    assert _canonical_family_sets(actual) == _canonical_family_sets(expected)
    assert _canonical_family_sets(actual) == [("a", "b"), ("b", "c")]


def test_jax_dense_growth_deduplicates_duplicate_seed_families() -> None:
    pairs = pd.DataFrame(
        [
            _accepted_pair("a", "b", 0.97),
            _accepted_pair("a", "c", 0.94),
            _accepted_pair("b", "c", 0.91),
            _accepted_pair("a", "d", 0.40),
            _accepted_pair("b", "d", 0.39),
            _accepted_pair("c", "d", 0.38),
        ]
    )

    actual, metrics = builder._grow_complete_link_families_jax(pairs, two_image_score_threshold=0.90)
    expected = _reference_growth_sets(pairs, two_image_score_threshold=0.90)

    assert _canonical_family_sets(actual) == _canonical_family_sets(expected)
    assert _canonical_family_sets(actual).count(("a", "b", "c", "d")) == 1
    assert metrics["family_growth_unique_masks"] == 1


def test_jax_dense_growth_keeps_two_image_gates() -> None:
    below_threshold = pd.DataFrame([_accepted_pair("a", "b", 0.89)])
    low_spec = pd.DataFrame([_accepted_pair("a", "b", 0.95, specz_score=0.60)])
    high_sed_rms = pd.DataFrame([_accepted_pair("a", "b", 0.95, sed_rms=1.10)])

    for pairs in [below_threshold, low_spec, high_sed_rms]:
        families, metrics = builder._grow_complete_link_families_jax(pairs, two_image_score_threshold=0.90)
        assert families == []
        assert metrics["family_growth_unique_masks"] == 0


def test_jax_dense_growth_chunked_seed_batches_match_one_large_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = pd.DataFrame(
        [
            _accepted_pair("a", "b", 0.97),
            _accepted_pair("a", "c", 0.94),
            _accepted_pair("b", "c", 0.91),
            _accepted_pair("d", "e", 0.96),
            _accepted_pair("d", "f", 0.92),
            _accepted_pair("e", "f", 0.89),
        ]
    )

    monkeypatch.setattr(builder, "DEFAULT_FAMILY_GROWTH_MAX_BATCH_CELLS", 1)
    chunked, chunked_metrics = builder._grow_complete_link_families_jax(pairs, two_image_score_threshold=0.90)
    monkeypatch.setattr(builder, "DEFAULT_FAMILY_GROWTH_MAX_BATCH_CELLS", 10_000)
    batched, batched_metrics = builder._grow_complete_link_families_jax(pairs, two_image_score_threshold=0.90)

    assert _canonical_family_sets(chunked) == _canonical_family_sets(batched)
    assert chunked_metrics["family_growth_n_batches"] == len(pairs)
    assert chunked_metrics["family_growth_all_seed_batch"] is False
    assert batched_metrics["family_growth_n_batches"] == 1
    assert batched_metrics["family_growth_all_seed_batch"] is True


def test_jax_dense_growth_max_object_guard_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = pd.DataFrame(
        [
            _accepted_pair("a", "b", 0.97),
            _accepted_pair("b", "c", 0.93),
        ]
    )
    monkeypatch.setattr(builder, "DEFAULT_FAMILY_GROWTH_MAX_OBJECTS", 2)

    with pytest.raises(RuntimeError, match="dense JAX family-growth limit"):
        builder._grow_complete_link_families_jax(pairs, two_image_score_threshold=0.90)


def test_complete_linkage_does_not_add_incompatible_third_image() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row(
                "b",
                offset_arcsec=10.0,
                zspec=2.0,
                rank=3,
                confidence="secure",
                mags=[24.0, 24.5, 24.5, 24.5, 24.5, 24.5, 24.6, 24.7],
            ),
            _row(
                "c",
                offset_arcsec=20.0,
                zspec=2.0,
                rank=3,
                confidence="secure",
                mags=[20.0, 28.0, 20.0, 28.0, 24.2, 28.0, 20.0, 23.7],
            ),
        ]
    )

    families, _members, pairs, _manifest = builder.build_cluster_image_families(
        catalog,
        spec,
        pair_score_threshold=0.25,
        two_image_score_threshold=0.0,
    )

    assert not pairs.loc[pairs["left_object_id"].eq("a") & pairs["right_object_id"].eq("c")].empty
    assert (
        pairs.loc[pairs["left_object_id"].eq("a") & pairs["right_object_id"].eq("c")].iloc[0]["hard_reject_reason"]
        == "color_rms_too_large"
    )
    assert families.empty or int(families["n_images"].max()) < 3


def test_reported_family_spans_are_limited_to_default_max_span() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("b", offset_arcsec=20.0, zspec=2.001, rank=3, confidence="secure"),
            _row("c", offset_arcsec=40.0, zspec=1.999, rank=3, confidence="secure"),
            _row("too_far", offset_arcsec=140.0, zspec=2.0, rank=3, confidence="secure"),
        ]
    )

    families, members, pairs, manifest = builder.build_cluster_image_families(catalog, spec)

    assert not families.empty
    assert families["max_separation_kpc"].max() <= builder.DEFAULT_MAX_FAMILY_SPAN_KPC
    assert "too_far" not in set(members["object_id"])
    rejected = pairs.loc[pairs["right_object_id"].eq("too_far") | pairs["left_object_id"].eq("too_far")]
    assert rejected.empty
    assert manifest["max_family_span_kpc"] == builder.DEFAULT_MAX_FAMILY_SPAN_KPC
    assert manifest["family_growth_backend"] == "jax_dense"
    assert manifest["n_family_growth_objects"] >= 3
    assert manifest["n_family_growth_seed_edges"] >= 3
    assert manifest["family_growth_n_batches"] >= 1
    assert manifest["family_growth_unique_masks"] >= 1
    assert manifest["family_growth_packed_mask_bytes"] > 0


def test_image_family_candidates_are_limited_to_circular_default_fov() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    scale = builder.kpc_per_arcsec(spec.z_lens)
    inside_offset = 450.0 / scale
    diagonal_offset = 400.0 / scale
    center_member = _row("center-member", offset_arcsec=0.0, zspec=spec.z_lens, rank=3, confidence="secure")
    center_member["member_probability"] = 0.95
    center_member["member_for_lensing"] = True
    diagonal_outside = _row("diagonal-outside", offset_arcsec=diagonal_offset, zspec=2.0, rank=3, confidence="secure")
    diagonal_outside["dec"] = -diagonal_offset / 3600.0
    catalog = pd.DataFrame(
        [
            center_member,
            _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("b", offset_arcsec=inside_offset, zspec=2.001, rank=3, confidence="secure"),
            diagonal_outside,
        ]
    )

    families, members, _pairs, manifest = builder.build_cluster_image_families(
        catalog,
        spec,
        pair_score_threshold=0.0,
        two_image_score_threshold=0.0,
    )

    assert len(families) == 1
    assert set(members["object_id"]) == {"a", "b"}
    assert manifest["image_family_fov_kpc"] == builder.DEFAULT_IMAGE_FAMILY_FOV_KPC
    assert manifest["image_family_fov_shape"] == "circle"
    assert manifest["image_family_fov_radius_kpc"] == builder.DEFAULT_IMAGE_FAMILY_FOV_KPC / 2.0
    assert manifest["n_rejected_outside_family_fov"] == 1


def test_jax_family_manifest_reports_pair_pruning_counters() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    row_a = _row(
        "a",
        offset_arcsec=0.0,
        zspec=2.0,
        rank=3,
        confidence="secure",
        mags=[24.0, 24.2, 24.3, 24.2, 24.1, 23.9, 23.8, np.nan],
    )
    row_b = _row(
        "b",
        offset_arcsec=10.0,
        zspec=2.0,
        rank=3,
        confidence="secure",
        mags=[np.nan, 24.2, 24.3, 24.2, 24.1, 23.9, 23.8, 23.7],
    )
    row_a["mag_F140W"] = 23.6
    row_b["mag_F110W"] = 23.6
    catalog = pd.DataFrame([row_a, row_b])

    families, members, pairs, manifest = builder.build_cluster_image_families(catalog, spec)

    assert families.empty
    assert members.empty
    assert pairs.empty
    assert manifest["pair_score_backend"] == "jax"
    assert manifest["family_growth_backend"] == "jax_dense"
    assert manifest["n_spatial_pairs"] == 1
    assert manifest["n_pruned_common_bands"] == 1
    assert manifest["n_scored_pairs"] == 0
    assert manifest["n_family_growth_objects"] == 0
    assert manifest["n_family_growth_seed_edges"] == 0
    assert manifest["family_growth_n_batches"] == 0
    assert manifest["family_growth_unique_masks"] == 0
    assert manifest["family_growth_packed_mask_bytes"] == 0
    assert manifest["family_growth_all_seed_batch"] is False


def test_two_image_families_require_strict_evidence() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    photo_catalog = pd.DataFrame(
        [
            _row("p1", offset_arcsec=0.0, zphot=2.0, zphot_low=1.9, zphot_high=2.1),
            _row("p2", offset_arcsec=10.0, zphot=2.0, zphot_low=1.9, zphot_high=2.1),
        ]
    )
    spec_catalog = pd.DataFrame(
        [
            _row("s1", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("s2", offset_arcsec=10.0, zspec=2.0, rank=3, confidence="secure"),
        ]
    )

    photo_families, _photo_members, _photo_pairs, _photo_manifest = builder.build_cluster_image_families(
        photo_catalog,
        spec,
    )
    spec_families, spec_members, _spec_pairs, _spec_manifest = builder.build_cluster_image_families(
        spec_catalog,
        spec,
    )

    assert photo_families.empty
    assert len(spec_families) == 1
    assert set(spec_members["object_id"]) == {"s1", "s2"}


def test_three_image_family_can_include_qualified_photoz_companion_with_two_specz_anchors() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("s1", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("s2", offset_arcsec=10.0, zspec=2.002, rank=3, confidence="secure"),
            _row("photo", offset_arcsec=20.0, zphot=2.01, zphot_low=1.8, zphot_high=2.2),
        ]
    )

    families, members, pairs, _manifest = builder.build_cluster_image_families(
        catalog,
        spec,
    )

    assert len(families) == 1
    assert int(families.iloc[0]["n_images"]) == 3
    assert "incomplete_specz" in families.iloc[0]["review_flags"]
    assert set(members["object_id"]) == {"s1", "s2", "photo"}
    assert set(pairs["redshift_relation"]) == {"both_specz", "specz_photoz"}


def test_single_specz_anchor_can_form_three_image_photoz_family() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("spec", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("p1", offset_arcsec=10.0, zphot=2.02, zphot_low=1.8, zphot_high=2.2),
            _row("p2", offset_arcsec=20.0, zphot=2.01, zphot_low=1.8, zphot_high=2.2),
        ]
    )

    families, members, pairs, manifest = builder.build_cluster_image_families(catalog, spec)

    assert len(families) == 1
    assert int(families.iloc[0]["n_images"]) == 3
    assert families.iloc[0]["family_z_method"] == "single_specz_anchor"
    assert "single_specz_anchor" in families.iloc[0]["review_flags"]
    assert (
        "photo_photo_diagnostic_only" in families.iloc[0]["review_flags"]
        or "photo_photo_complete_link" in families.iloc[0]["review_flags"]
    )
    assert set(members["object_id"]) == {"spec", "p1", "p2"}
    assert pairs.loc[pairs["redshift_relation"].eq("photoz_only"), "hard_reject_reason"].eq("").all()
    assert manifest["n_single_specz_anchor_families"] == 1


def test_single_specz_anchor_family_flags_near_limit_photoz_delta() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("spec", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("p1", offset_arcsec=10.0, zphot=2.00, zphot_low=1.8, zphot_high=2.2),
            _row("p2", offset_arcsec=20.0, zphot=2.85, zphot_low=1.8, zphot_high=3.0),
        ]
    )

    families, members, _pairs, manifest = builder.build_cluster_image_families(catalog, spec)

    assert len(families) == 1
    assert "photoz_delta_near_limit" in families.iloc[0]["review_flags"]
    assert set(members["object_id"]) == {"spec", "p1", "p2"}
    assert manifest["family_photoz_delta_max"] == builder.DEFAULT_FAMILY_PHOTOZ_DELTA_MAX


def test_single_specz_anchor_family_rejects_large_companion_photoz_delta() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("spec", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("p1", offset_arcsec=10.0, zphot=2.00, zphot_low=1.8, zphot_high=2.2),
            _row("p2", offset_arcsec=20.0, zphot=3.10, zphot_low=1.8, zphot_high=3.2),
        ]
    )

    families, members, pairs, manifest = builder.build_cluster_image_families(catalog, spec)

    assert families.empty
    assert members.empty
    assert not pairs.loc[pairs["redshift_relation"].eq("specz_photoz")].empty
    assert manifest["n_single_specz_anchor_families"] == 0


def test_single_specz_anchor_plus_one_photoz_does_not_form_two_image_family() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("spec", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("p1", offset_arcsec=10.0, zphot=2.02, zphot_low=1.8, zphot_high=2.2),
        ]
    )

    families, members, _pairs, manifest = builder.build_cluster_image_families(catalog, spec)

    assert families.empty
    assert members.empty
    assert manifest["n_single_specz_anchor_families"] == 0


def test_single_specz_anchor_rejects_inconsistent_photoz_companion() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("spec", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("good", offset_arcsec=10.0, zphot=2.02, zphot_low=1.8, zphot_high=2.2),
            _row("bad", offset_arcsec=20.0, zphot=3.2, zphot_low=3.0, zphot_high=3.4),
        ]
    )

    families, members, pairs, _manifest = builder.build_cluster_image_families(catalog, spec)

    assert families.empty
    assert members.empty
    bad_pair = pairs.loc[pairs["left_object_id"].eq("spec") & pairs["right_object_id"].eq("bad")]
    assert not bad_pair.empty
    assert bad_pair.iloc[0]["hard_reject_reason"] == "photoz_inconsistent"


def test_single_specz_anchor_family_preserves_max_span() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    offset = 350.0 / builder.kpc_per_arcsec(spec.z_lens)
    catalog = pd.DataFrame(
        [
            _row("spec", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("p1", offset_arcsec=offset, zphot=2.02, zphot_low=1.8, zphot_high=2.2),
            _row("p2", offset_arcsec=-offset, zphot=2.01, zphot_low=1.8, zphot_high=2.2),
        ]
    )

    families, members, _pairs, _manifest = builder.build_cluster_image_families(catalog, spec)

    assert families.empty
    assert members.empty


def test_family_summary_uses_any_finite_specz() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("secure", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure", zphot=0.2),
            _row("low", offset_arcsec=10.0, zspec=2.001, rank=0.0, confidence="low_quality", zphot=5.0),
        ]
    )

    families, members, _pairs, manifest = builder.build_cluster_image_families(
        catalog,
        spec,
        pair_score_threshold=0.25,
        two_image_score_threshold=0.0,
    )

    assert len(families) == 1
    np.testing.assert_allclose(families.iloc[0]["family_z_best"], 2.0005)
    assert families.iloc[0]["family_z_method"] == "specz_median"
    assert set(members["object_id"]) == {"secure", "low"}


def test_family_summary_allows_three_image_photoz_only_family() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("p1", offset_arcsec=0.0, zphot=2.00),
            _row("p2", offset_arcsec=10.0, zphot=2.02),
            _row("p3", offset_arcsec=20.0, zphot=2.01),
        ]
    )

    families, members, _pairs, manifest = builder.build_cluster_image_families(
        catalog,
        spec,
        pair_score_threshold=0.25,
        two_image_score_threshold=0.0,
    )

    assert len(families) == 1
    assert set(members["object_id"]) == {"p1", "p2", "p3"}
    assert families.iloc[0]["family_z_method"] == "photoz_median"
    assert "photoz_only_anchor" in families.iloc[0]["review_flags"]
    assert "missing_specz" in families.iloc[0]["review_flags"]
    assert families.iloc[0]["family_probability"] <= builder.DEFAULT_PHOTOZ_ONLY_FAMILY_PROBABILITY_CAP
    assert manifest["n_rejected_missing_strong_specz"] == 0
    assert manifest["n_image_family_photoz_companion_candidates"] == 3
    assert manifest["n_photoz_only_families"] == 1


def test_photoz_only_family_requires_complete_pairwise_photoz_consistency() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("p1", offset_arcsec=0.0, zphot=2.00),
            _row("p2", offset_arcsec=10.0, zphot=2.02),
            _row("bad", offset_arcsec=20.0, zphot=3.10),
        ]
    )

    families, members, pairs, manifest = builder.build_cluster_image_families(
        catalog,
        spec,
        pair_score_threshold=0.25,
        two_image_score_threshold=0.0,
    )

    assert families.empty
    assert members.empty
    assert "photoz_delta_too_large" in set(pairs["hard_reject_reason"])
    assert manifest["n_photoz_only_families"] == 0


def test_membership_probabilities_are_normalized_per_object() -> None:
    members = pd.DataFrame(
        {
            "object_id": ["a", "a", "b"],
            "raw_probability": [0.8, 0.7, 0.4],
            "membership_probability": [0.8, 0.7, 0.4],
        }
    )

    normalized = builder.normalize_membership_probabilities(members)

    np.testing.assert_allclose(normalized.loc[normalized["object_id"] == "a", "membership_probability"].sum(), 1.0)
    np.testing.assert_allclose(normalized.loc[normalized["object_id"] == "b", "membership_probability"].sum(), 0.4)


def test_existing_catalog_rows_do_not_override_specz_conflict() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _row("a", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure"),
            _row("b", offset_arcsec=10.0, zspec=2.02, rank=3, confidence="secure"),
        ]
    )

    families, members, pairs, _manifest = builder.build_cluster_image_families(catalog, spec)

    assert families.empty
    assert members.empty
    assert pairs.iloc[0]["hard_reject_reason"] == "secure_or_probable_specz_conflict"


def test_refined_member_probability_excludes_family_candidate() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    likely_member = _row("member", offset_arcsec=0.0, zspec=2.0, rank=3, confidence="secure")
    likely_member["member_probability"] = 0.9
    likely_member["member_for_lensing"] = True
    background = _row("background", offset_arcsec=10.0, zspec=2.0, rank=3, confidence="secure")
    background["member_probability"] = 0.0
    background["member_for_lensing"] = False
    catalog = pd.DataFrame([likely_member, background])

    candidates = builder.prepare_candidates(catalog, spec)

    assert candidates["object_id"].tolist() == ["background"]
