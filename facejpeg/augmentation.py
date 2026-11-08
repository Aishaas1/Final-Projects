"""A shared JPEG recipe for training images and cached CPU feature views."""
import math
from numbers import Integral, Real

from .common import canonical, jpeg_copy


def jpeg_settings(qualities=(), probability=.5):
    if not isinstance(qualities, (list, tuple)) or any(
        not isinstance(q, Integral) or isinstance(q, bool) or not 1 <= q <= 100 for q in qualities
    ):
        raise ValueError("jpeg_qualities must be a list/tuple of integers in [1,100]")
    if not isinstance(probability, Real) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("jpeg_probability must be finite and in [0,1]")
    qualities = tuple(dict.fromkeys(int(q) for q in qualities))
    return qualities, float(probability) if qualities else 0.


def jpeg_training_view(image, quality, preprocess):
    # Encoding follows the same canonical-then-JPEG protocol as test/inference.
    return jpeg_copy(canonical(image, preprocess), quality)
