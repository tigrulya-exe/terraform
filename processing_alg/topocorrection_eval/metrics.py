from dataclasses import dataclass
from typing import Any

import numpy as np

from processing_alg.topocorrection_eval.eval import EvaluationAlgorithm


def minmax(values: list):
    min_val = values[0]
    max_val = values[0]
    for i in values:
        if i < min_val:
            min_val = i
        if i > max_val:
            max_val = i

    return min_val, max_val


# тут надо наверн передавать не сами ориг байтс, а геттер хз
@dataclass
class EvalContext:
    orig_band: EvaluationAlgorithm.BandInfo
    current_band: EvaluationAlgorithm.BandInfo

    def current_minmax(self):
        return self.band_minmax(self.orig_band)

    def orig_minmax(self):
        return self.band_minmax(self.orig_band)

    def band_minmax(self, band):
        band_min, band_max, *_ = band.gdal_band.GetStatistics(True, True)
        return band_min, band_max


class EvalMetric:
    def __init__(self, is_reduction=True, weight=1.0):
        self.weight = weight
        self.norm_func = EvalMetric._revert_norm if is_reduction else EvalMetric._norm
        self.ctx = None

    def id(self):
        pass

    def name(self):
        pass

    def evaluate(self, values: list, ctx: EvalContext) -> float:
        pass

    def combine(self, original: float, corrected: float):
        return corrected - original

    # norm all values to [0, 1] range
    def norm(self, metrics: list[float]) -> list[float]:
        min_val, max_val = minmax(metrics)
        return [self.norm_func(metric, min_val, max_val) for metric in metrics]

    @staticmethod
    def _norm(metric, min_val, max_val):
        return (metric - min_val) / (max_val - min_val)

    @staticmethod
    def _revert_norm(metric, min_val, max_val):
        return 1 - EvalMetric._norm(metric, min_val, max_val)


class StdMetric(EvalMetric):
    def id(self):
        return "std_reduction"

    def name(self):
        return "Mean reflectance reduction"

    def evaluate(self, values) -> float:
        return np.std(values)


class CvMetric(EvalMetric):
    def id(self):
        return "cv_reduction"

    def name(self):
        return "Coefficient of variation reduction"

    def evaluate(self, values: list):
        return np.std(values) / np.mean(values)


# todo add iqr outliers metric https://www.scribbr.com/statistics/outliers/
class InterQuartileRangeMetric(EvalMetric):
    def id(self):
        return "iqr_reduction"

    def name(self):
        return "Inter quartile range reduction"

    @staticmethod
    # @functools.cache
    def get_q1_q3(values):
        return np.percentile(values, [25, 75])

    def evaluate(self, values: list):
        q1, q3 = InterQuartileRangeMetric.get_q1_q3(values)
        return q3 - q1


class RelativeMedianDifferenceRangeMetric(EvalMetric):
    def id(self):
        return "relative_median_difference"

    def name(self):
        return "Relative median difference"

    # @functools.cache
    def evaluate(self, values: list):
        return np.median(values)

    def combine(self, original: float, corrected: float):
        return abs(corrected - original) / original


class OutliersCountMetric(EvalMetric):
    def __init__(self):
        super().__init__(is_reduction=False)

    def evaluate(self, values) -> float:
        return np.count_nonzero(self._outliers_filter(values))

    def _outliers_filter(self, values):
        pass


class ThresholdOutliersCountMetric(OutliersCountMetric):
    def __init__(self, min_threshold=None, max_threshold=None):
        super().__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def init(self, ctx: EvalContext):
        super().init(ctx)
        orig_min, orig_max = self.ctx.orig_minmax()
        self.min_threshold = self.min_threshold or orig_min
        self.max_threshold = self.max_threshold or orig_max

    def _outliers_filter(self, values):
        return np.logical_and(self.min_threshold < values, values < self.max_threshold)


class IqrOutliersCountMetric(OutliersCountMetric):
    def _outliers_filter(self, values):
        q1, q3 = np.percentile(values, [25, 75])
        min_threshold = q1 - (q3 - q1)
        max_threshold = q3 + (q3 - q1)
        return np.logical_and(min_threshold < values, values < max_threshold)


class RegressionSlopeMetric(EvalMetric):
    def evaluate(self, values: list) -> float:
        _, slope = np.polynomial.polynomial.polyfit(values, self.ctx.orig_bytes, 1)
        return abs(slope)

