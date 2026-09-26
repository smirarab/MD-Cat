"""Affine time coordinates used by fitting and confidence-interval solvers."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TimeScale:
    # Identity is also the coordinate system of legacy CI checkpoints.
    origin: float = 0.0
    span: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.origin) or not math.isfinite(self.span) or self.span <= 0:
            raise ValueError('time normalization requires a finite origin and positive finite span')

    @classmethod
    def from_sampling_times(cls, times):
        values = list(times.values())
        if not values or not all(math.isfinite(t) for t in values):
            raise ValueError('calibration times must be finite and nonempty')
        origin = min(values)
        span = max(values) - origin
        if not math.isfinite(span) or span <= 0:
            raise ValueError('dating requires at least two distinct calibration times with a finite span')
        return cls(origin, span)

    def normalize_times(self, times):
        return {name: (t - self.origin) / self.span for name, t in times.items()}

    def restore_times(self, times):
        return {name: self.origin + t * self.span for name, t in times.items()}
