import math

import numpy as np
import pytest

from arctrace import geometry
from arctrace_synth import make_tan_wcs
from lenscluster.lenstool_parser import _fallback_radec_to_offsets, _offsets_to_radec


def test_wrap_axial_edges() -> None:
    assert geometry.wrap_axial(0.0) == pytest.approx(0.0)
    assert geometry.wrap_axial(math.pi) == pytest.approx(0.0, abs=1e-12)
    assert geometry.wrap_axial(-1.0e-9) == pytest.approx(math.pi - 1.0e-9, abs=1e-12)
    assert geometry.wrap_axial(2.0 * math.pi + 0.3) == pytest.approx(0.3)
    values = geometry.wrap_axial(np.array([0.0, math.pi, 1.5 * math.pi, -0.25 * math.pi]))
    np.testing.assert_allclose(values, [0.0, 0.0, 0.5 * math.pi, 0.75 * math.pi], atol=1e-12)


def test_axial_difference_wraps() -> None:
    assert geometry.axial_difference(0.1, math.pi + 0.05) == pytest.approx(0.05)
    assert geometry.axial_difference(math.pi - 0.01, 0.01) == pytest.approx(-0.02)


@pytest.mark.parametrize(
    ("theta_eofn", "expected"),
    [
        (0.0, 0.5 * math.pi),  # North
        (0.5 * math.pi, 0.0),  # East (axis along -x is the same axial angle as +x)
        (0.25 * math.pi, 0.75 * math.pi),
        (5.0 * math.pi / 6.0, math.pi / 3.0),
    ],
)
def test_position_angle_to_offset_frame_angle(theta_eofn: float, expected: float) -> None:
    assert geometry.position_angle_to_offset_frame_angle(theta_eofn) == pytest.approx(expected, abs=1e-12)


def test_radec_to_solver_offsets_matches_lenstool_parser() -> None:
    rng = np.random.default_rng(7)
    ra0, dec0 = 39.97, -1.58
    ra = ra0 + rng.uniform(-0.05, 0.05, size=20)
    dec = dec0 + rng.uniform(-0.05, 0.05, size=20)
    x_ours, y_ours = geometry.radec_to_solver_offsets(ra, dec, ra0, dec0)
    _, _, x_ref, y_ref = _fallback_radec_to_offsets(ra, dec, ra0, dec0)
    np.testing.assert_allclose(x_ours, x_ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(y_ours, y_ref, rtol=0, atol=1e-9)


def test_solver_offsets_round_trip_with_parser_inverse() -> None:
    ra0, dec0 = 110.3, 35.2
    for x, y in [(12.3, -4.5), (-80.0, 33.3), (0.0, 0.0)]:
        ra, dec = geometry.solver_offsets_to_radec(x, y, ra0, dec0)
        ra_ref, dec_ref = _offsets_to_radec(x, y, ra0, dec0)
        assert float(ra) == pytest.approx(ra_ref, abs=1e-12)
        assert float(dec) == pytest.approx(dec_ref, abs=1e-12)
        x_back, y_back = geometry.radec_to_solver_offsets(float(ra), float(dec), ra0, dec0)
        assert float(x_back) == pytest.approx(x, abs=1e-9)
        assert float(y_back) == pytest.approx(y, abs=1e-9)


def test_pole_proximity_raises() -> None:
    with pytest.raises(ValueError):
        geometry.radec_to_solver_offsets(10.0, 89.9999999, 10.0, 89.9999999)


def test_pixel_tangent_north_up_flipped_wcs() -> None:
    # CDELT1 < 0: pixel +x points West, +y points North -> pixel angle equals
    # the offsets-frame angle.
    wcs = make_tan_wcs(39.97, -1.58, pixscale_arcsec=0.06, rotation_deg=0.0, flip_ra=True)
    angle = geometry.pixel_tangent_to_offset_angle(wcs, 50.0, 50.0, (1.0, 1.0))
    assert angle == pytest.approx(0.25 * math.pi, abs=2e-4)


def test_pixel_tangent_unflipped_wcs_detects_parity() -> None:
    # CDELT1 > 0: pixel +x points East; a NE pixel diagonal is NW on the sky's
    # offsets frame.
    wcs = make_tan_wcs(39.97, -1.58, pixscale_arcsec=0.06, rotation_deg=0.0, flip_ra=False)
    angle = geometry.pixel_tangent_to_offset_angle(wcs, 50.0, 50.0, (1.0, 1.0))
    assert angle == pytest.approx(0.75 * math.pi, abs=2e-4)
    angle_x = geometry.pixel_tangent_to_offset_angle(wcs, 50.0, 50.0, (1.0, 0.0))
    assert min(angle_x, math.pi - angle_x) == pytest.approx(0.0, abs=2e-4)


@pytest.mark.parametrize("rotation_deg", [0.0, 30.0, -75.0])
@pytest.mark.parametrize("flip_ra", [True, False])
def test_pixel_tangent_matches_brute_force(rotation_deg: float, flip_ra: bool) -> None:
    wcs = make_tan_wcs(200.5, 12.4, pixscale_arcsec=0.05, rotation_deg=rotation_deg, flip_ra=flip_ra)
    ra0, dec0 = 200.5, 12.4
    rng = np.random.default_rng(3)
    for _ in range(5):
        x0, y0 = rng.uniform(30.0, 70.0, size=2)
        direction = rng.normal(size=2)
        angle = geometry.pixel_tangent_to_offset_angle(wcs, x0, y0, (direction[0], direction[1]))
        # Brute force: displace in pixel space, convert both endpoints through
        # the WCS to the offsets frame, take atan2 of the difference.
        unit = direction / np.hypot(*direction)
        step = 0.5
        xs = np.array([x0, x0 + step * unit[0]])
        ys = np.array([y0, y0 + step * unit[1]])
        x_off, y_off = geometry.pixel_points_to_offset_frame(wcs, xs, ys, ra0, dec0)
        brute = geometry.wrap_axial(math.atan2(y_off[1] - y_off[0], x_off[1] - x_off[0]))
        assert abs(geometry.axial_difference(angle, brute)) < 1e-5
