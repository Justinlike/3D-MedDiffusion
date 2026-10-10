"""Shared preprocessing helpers for volumetric NIfTI datasets."""

from collections.abc import Sequence

import torch
import torchio as tio


DEFAULT_TARGET_SHAPE = (512, 512, 64)
DEFAULT_INTENSITY_NORMALIZATION = "minmax"
SUPPORTED_INTENSITY_NORMALIZATIONS = {"minmax", "none"}


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


def normalize_volume(
    image: tio.ScalarImage,
    mode: str = DEFAULT_INTENSITY_NORMALIZATION,
    eps: float = 1e-6,
) -> tio.ScalarImage:
    """Normalize an intensity image to ``[0, 1]``.

    ``minmax`` follows the per-volume normalization used by the NCCT2CCTA
    reference pipeline. ``none`` is intended for data that has already been
    normalized and therefore validates, rather than silently accepting, the
    input range.
    """
    mode = str(mode).lower()
    if mode not in SUPPORTED_INTENSITY_NORMALIZATIONS:
        raise ValueError(
            f"intensity_normalization must be one of "
            f"{sorted(SUPPORTED_INTENSITY_NORMALIZATIONS)}, got {mode!r}"
        )

    data = image.data.to(torch.float32).clone()
    if not torch.isfinite(data).all():
        raise ValueError("input volume contains NaN or infinite intensities")

    minimum = data.amin()
    maximum = data.amax()
    if mode == "minmax":
        value_range = maximum - minimum
        if value_range <= eps:
            data = torch.zeros_like(data)
        else:
            data = (data - minimum) / value_range
    elif minimum < -eps or maximum > 1.0 + eps:
        raise ValueError(
            "intensity_normalization='none' requires input data in [0, 1], "
            f"but observed [{minimum.item():.6g}, {maximum.item():.6g}]"
        )

    return tio.ScalarImage(tensor=data.clamp_(0.0, 1.0), affine=image.affine)


def preprocess_volume(
    image: tio.ScalarImage,
    target_shape: Sequence[int] | None,
    intensity_normalization: str = DEFAULT_INTENSITY_NORMALIZATION,
) -> tio.ScalarImage:
    """Resize a volume and normalize its intensities for VAE input."""
    image = resize_volume(image, target_shape)
    return normalize_volume(image, intensity_normalization)
