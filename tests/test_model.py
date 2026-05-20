from dataclasses import dataclass

import numpy as np

from lenscluster.model import latent_array_to_physical, latent_to_physical, physical_to_latent


@dataclass(frozen=True)
class TransformSpec:
    transform_kind: str
    transform_offset: float = 0.0
    transform_scale: float = 1.0


def test_positive_log_transform_round_trip() -> None:
    spec = TransformSpec("log_positive")

    latent = physical_to_latent(12.5, spec)

    np.testing.assert_allclose(latent_to_physical(latent, spec), 12.5)


def test_offset_log_transform_round_trip_array() -> None:
    spec = TransformSpec("log_offset_positive", transform_offset=2.0)
    latent = np.asarray([physical_to_latent(3.0, spec), physical_to_latent(6.0, spec)])

    physical = latent_array_to_physical(latent, spec)

    np.testing.assert_allclose(physical, np.asarray([3.0, 6.0]))


def test_affine_transform_round_trip_array() -> None:
    spec = TransformSpec("affine", transform_offset=0.25, transform_scale=0.3)
    latent = np.asarray([0.0, 1.0, -2.0])

    physical = latent_array_to_physical(latent, spec)

    np.testing.assert_allclose(physical, np.asarray([0.25, 0.55, -0.35]))
    np.testing.assert_allclose([physical_to_latent(value, spec) for value in physical], latent)
