"""Shared preprocessing helpers for volumetric NIfTI datasets."""

from collections.abc import Sequence

import torchio as tio


DEFAULT_TARGET_SHAPE = (512, 512, 64)


def normalize_target_shape(target_shape: Sequence[int] | None) -> tuple[int, int, int] | None:
    """Validate and normalize a spatial target shape.

    ``None`` disables resizing.  TorchIO uses spatial order ``(W, H, D)`` for
    image shapes, matching the NIfTI loading convention used by this project.
    """
    if target_shape is None:
        return None
    shape = tuple(int(size) for size in target_shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"target_shape must contain three positive integers, got {target_shape!r}")
    return shape


def resize_volume(image: tio.ScalarImage, target_shape: Sequence[int] | None) -> tio.ScalarImage:
    """Resample ``image`` to ``target_shape`` when its spatial shape differs."""
    shape = normalize_target_shape(target_shape)
    if shape is None or tuple(image.spatial_shape) == shape:
        return image
    return tio.Resize(shape)(image)
