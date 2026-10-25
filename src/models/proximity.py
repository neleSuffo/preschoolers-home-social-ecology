"""
src/models/proximity.py

Estimates normalized physical proximity (0.0 to 1.0) from bounding box geometry
using logarithmic area scaling and aspect-ratio anomaly handling.
"""

import logging
from typing import Optional, Sequence
import numpy as np

from src.constants import Proximity

logger = logging.getLogger(__name__)


class Proximity:
    """
    Computes normalized face proximity from bounding box area and aspect ratio.

    Applies a logarithmic scaling between empirical minimum and maximum
    face bounding box areas, with aspect-ratio deviation gating for partially
    occluded, close-up faces.
    """

    def __init__(self) -> None:
        refs = Proximity.REFERENCE_VALUES
        self.child_ref_close: float = refs["child_ref_close"]
        self.child_ref_far: float = refs["child_ref_far"]
        self.adult_ref_close: float = refs["adult_ref_close"]
        self.adult_ref_far: float = refs["adult_ref_far"]
        self.child_ref_aspect_ratio: float = refs["child_ref_aspect_ratio"]
        self.adult_ref_aspect_ratio: float = refs["adult_ref_aspect_ratio"]

        # Precompute logarithmic scaling denominators
        self.child_log_far: float = float(np.log(self.child_ref_far))
        self.child_log_range: float = float(np.log(self.child_ref_close) - self.child_log_far)
        self.adult_log_far: float = float(np.log(self.adult_ref_far))
        self.adult_log_range: float = float(np.log(self.adult_ref_close) - self.adult_log_far)

    def calculate(
        self,
        bbox: Sequence[float],
        class_id: int,
        aspect_ratio_threshold: float = Proximity.ASPECT_RATIO_THRESHOLD
    ) -> Optional[float]:
        """
        Calculate normalized physical proximity for a single bounding box.

        Parameters
        ----------
        bbox : Sequence[float]
            Bounding box coordinates as (x1, y1, x2, y2).
        class_id : int
            Target demographic class: 0 for child, 1 for adult.
        aspect_ratio_threshold : float, default=1.0
            Maximum allowed deviation (|observed - reference|) before
            treating an anomalous aspect ratio as a close-range partial face cut-off.

        Returns
        -------
        Optional[float]
            Normalized proximity value bounded between 0.0 and 1.0,
            or None if the class_id is invalid.
        """
        x1, y1, x2, y2 = map(float, bbox)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        if height <= 0.0 or width <= 0.0:
            return 0.0

        aspect_ratio = width / height

        if class_id == 0:
            ref_close = self.child_ref_close
            ref_far = self.child_ref_far
            ref_ar = self.child_ref_aspect_ratio
            log_far = self.child_log_far
            log_range = self.child_log_range
        elif class_id == 1:
            ref_close = self.adult_ref_close
            ref_far = self.adult_ref_far
            ref_ar = self.adult_ref_aspect_ratio
            log_far = self.adult_log_far
            log_range = self.adult_log_range
        else:
            logger.error(f"Invalid class_id: {class_id}. Must be 0 (child) or 1 (adult).")
            return None

        # Partial face check: extreme aspect ratio deviations close to the camera frame
        if abs(aspect_ratio - ref_ar) > aspect_ratio_threshold:
            logger.debug("Partial face detected: returning maximum proximity (1.0).")
            return 1.0

        # Boundary saturation limits
        if area >= ref_close:
            return 1.0
        if area <= ref_far:
            return 0.0

        # Logarithmic normalization
        proximity = (np.log(area) - log_far) / log_range
        return float(np.clip(proximity, 0.0, 1.0))


# Module-level instance to prevent repeated instantiations across millions of frames
_PROXIMITY_INSTANCE = Proximity()


def calculate_proximity(
    bbox: Sequence[float],
    class_id: int,
    aspect_ratio_threshold: float = Proximity.ASPECT_RATIO_THRESHOLD,
) -> Optional[float]:
    """
    Calculate proximity for a single bounding box using the cached calculator.

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding box coordinates as (x1, y1, x2, y2).
    class_id : int
        Class ID: 0 (child) or 1 (adult).
    aspect_ratio_threshold : float, default=1.0
        Maximum allowed deviation from reference aspect ratio.

    Returns
    -------
    Optional[float]
        Normalized proximity value in [0.0, 1.0].
    """
    return _PROXIMITY_INSTANCE.calculate(
        bbox=bbox,
        class_id=class_id,
        aspect_ratio_threshold=aspect_ratio_threshold,
    )