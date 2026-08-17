"""Compatibility namespace for the SleepWM research package.

The historical ``uniphysio_wm`` import path is retained so released checkpoints
and experiment scripts remain loadable. New public code may import the same API
from ``sleepwm``.
"""

from .models import (
    CausalPhysioWorldModel,
    MaskedMultiModalModel,
    MultiModalEncoder,
    ObservationConfig,
    SleepStageClassifier,
    TCNSleepClassifier,
)

__all__ = [
    "CausalPhysioWorldModel",
    "MaskedMultiModalModel",
    "MultiModalEncoder",
    "ObservationConfig",
    "SleepStageClassifier",
    "TCNSleepClassifier",
]

__version__ = "1.0.0"
