import pytest

from arctrace.errors import RegionParseError
from arctrace.regions import normalize_arc_id, parse_ds9_regions


def _write(tmp_path, text: str):
    path = tmp_path / "seeds.reg"
    path.write_text(text)
    return path


def test_parse_bergamini_style_ellipses(tmp_path) -> None:
    path = _write(
        tmp_path,
        "# Region file format: DS9 version 4.1\n"
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1\n'
        "fk5\n"
        'ellipse(342.19559,-44.528389,0.500",0.500",90) # color=red font="helvetica 14 normal" text={2a}\n'
        'ellipse(342.19483,-44.52735,0.500",0.500",90) # text={2b}\n',
    )
    seeds = parse_ds9_regions(path)
    assert len(seeds) == 2
    assert seeds[0].ra_deg == pytest.approx(342.19559)
    assert seeds[0].dec_deg == pytest.approx(-44.528389)
    assert seeds[0].label_raw == "2a"
    assert seeds[0].radius_arcsec == pytest.approx(0.5)
    assert seeds[1].label_raw == "2b"


def test_parse_points_circles_and_sexagesimal(tmp_path) -> None:
    path = _write(
        tmp_path,
        "icrs\n"
        "point(39.971234,-1.582345) # point=cross text={1.a}\n"
        "circle(02:39:53.10,-01:34:55.7,1.5\") # text={M3c}\n"
        "fk5; point(40.0,-1.6) # text={4.b}\n",
    )
    seeds = parse_ds9_regions(path)
    assert len(seeds) == 3
    assert seeds[0].label_raw == "1.a"
    # 02:39:53.10 hours = 39.97125 deg
    assert seeds[1].ra_deg == pytest.approx(39.97125, abs=1e-5)
    assert seeds[1].dec_deg == pytest.approx(-(1.0 + 34.0 / 60.0 + 55.7 / 3600.0), abs=1e-6)
    assert seeds[1].radius_arcsec == pytest.approx(1.5)
    assert seeds[2].label_raw == "4.b"


def test_missing_frame_errors(tmp_path) -> None:
    path = _write(tmp_path, "point(10.0,1.0) # text={1.a}\n")
    with pytest.raises(RegionParseError, match="frame"):
        parse_ds9_regions(path)


def test_image_frame_rejected(tmp_path) -> None:
    path = _write(tmp_path, "image\npoint(100,200) # text={1.a}\n")
    with pytest.raises(RegionParseError, match="unsupported"):
        parse_ds9_regions(path)


def test_empty_file_errors(tmp_path) -> None:
    path = _write(tmp_path, "fk5\n# nothing here\n")
    with pytest.raises(RegionParseError, match="No usable"):
        parse_ds9_regions(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2a", "2a"),
        ("2.a", "2.a"),
        ("M12b", "M12b"),
        ("m12b", "m12b"),
        ("14C.d", "14C.d"),
        ("a.B", "a.B"),
    ],
)
def test_normalize_arc_id(raw: str, expected: str) -> None:
    assert normalize_arc_id(raw) == expected


@pytest.mark.parametrize("raw", ["", " ", "arc id", "1\t2"])
def test_normalize_arc_id_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_arc_id(raw)
