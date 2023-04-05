from dataclasses import dataclass

import numpy as np
from pandas import DataFrame

from ...processing_alg.topocorrection_eval.eval import EvaluationAlgorithm


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
    current_band: EvaluationAlgorithm.BandInfo
    orig_band: EvaluationAlgorithm.BandInfo
    orig_metrics: dict[str, float]

    def orig_minmax(self):
        return self.band_minmax(self.orig_band)

    def band_minmax(self, band):
        band_min, band_max, *_ = band.gdal_band.GetStatistics(True, True)
        return band_min, band_max

    def orig_metric(self, metric) -> float:
        return self.orig_metrics[metric.id()]


class EvalMetric:
    def __init__(self, is_reduction=True, weight=1.0):
        self.weight = weight
        self.is_reduction = is_reduction

    @staticmethod
    def id():
        pass

    @staticmethod
    def name():
        pass

    def unary(self, values: list) -> float:
        pass

    def binary(self, values: list, ctx: EvalContext) -> float:
        return self.combine(self.unary(values), ctx.orig_metric(self))

    def combine(self, original: float, corrected: float):
        return corrected if original == 0 else (corrected - original) / original

    # norm all values to [0, 1] range
    def norm(self, metrics: DataFrame) -> DataFrame:
        metrics_min = metrics.min()
        normalized_df = (metrics - metrics_min) / (metrics.max() - metrics_min)
        return 1 - normalized_df if self.is_reduction else normalized_df


class StdMetric(EvalMetric):
    @staticmethod
    def id():
        return "std_reduction"

    @staticmethod
    def name():
        return "Mean reflectance reduction"

    def unary(self, values: list) -> float:
        return np.std(values)


class CvMetric(EvalMetric):
    @staticmethod
    def id():
        return "cv_reduction"

    @staticmethod
    def name():
        return "Coefficient of variation reduction"

    def unary(self, values: list) -> float:
        return np.std(values) / np.mean(values)


# todo add iqr outliers metric https://www.scribbr.com/statistics/outliers/
class InterQuartileRangeMetric(EvalMetric):
    @staticmethod
    def id():
        return "iqr_reduction"

    @staticmethod
    def name():
        return "Inter quartile range reduction"

    @staticmethod
    # @functools.cache
    def get_q1_q3(values):
        return np.percentile(values, [25, 75])

    def unary(self, values: list) -> float:
        q1, q3 = InterQuartileRangeMetric.get_q1_q3(values)
        return q3 - q1


class RelativeMedianDifferenceRangeMetric(EvalMetric):
    @staticmethod
    def id():
        return "relative_median_difference"

    @staticmethod
    def name():
        return "Relative median difference"

    # @functools.cache
    def unary(self, values: list) -> float:
        return np.median(values)

    def combine(self, original: float, corrected: float):
        return abs(corrected) if original == 0 else abs(corrected - original) / original


class OutliersCountMetric(EvalMetric):
    def __init__(self):
        super().__init__(is_reduction=False)

    def combine(self, original: float, corrected: float):
        return corrected

    def unary(self, values: list) -> float:
        return 0

    def binary(self, values, ctx: EvalContext) -> float:
        return np.count_nonzero(self._outliers_filter(values, ctx))

    def _outliers_filter(self, values: list, ctx: EvalContext):
        pass


class ThresholdOutliersCountMetric(OutliersCountMetric):
    @staticmethod
    def id():
        return "outliers_threshold"

    @staticmethod
    def name():
        return "Number of outliers (static thresholds)"

    def __init__(self, min_threshold=None, max_threshold=None):
        super().__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def _init_thresholds(self, ctx: EvalContext):
        if self.min_threshold is None or self.max_threshold is None:
            orig_min, orig_max = ctx.orig_minmax()
            self.min_threshold = self.min_threshold or orig_min
            self.max_threshold = self.max_threshold or orig_max

    def _outliers_filter(self, values, ctx: EvalContext):
        self._init_thresholds(ctx)
        return np.logical_and(self.min_threshold >= values, values >= self.max_threshold)


class IqrOutliersCountMetric(OutliersCountMetric):
    @staticmethod
    def id():
        return "outliers_iqr"

    @staticmethod
    def name():
        return "Number of outliers - elements out of the [Q1 - IQR; Q3 + IQR] interval"

    def _outliers_filter(self, values, ctx: EvalContext):
        q1, q3 = np.percentile(values, [25, 75])
        min_threshold = q1 - (q3 - q1)
        max_threshold = q3 + (q3 - q1)
        return np.logical_and(min_threshold >= values, values >= max_threshold)


class RegressionSlopeMetric(EvalMetric):
    @staticmethod
    def id():
        return "correlation_regression_slope"

    @staticmethod
    def name():
        return "Slope of the regression between band values and solar incidence angle"

    def unary(self, values: list) -> float:
        return 0

    def combine(self, original: float, corrected: float):
        return corrected

    def binary(self, values: list, ctx: EvalContext) -> float:
        _, slope = np.polynomial.polynomial.polyfit(values, ctx.orig_band.bytes, 1)
        return abs(slope)


DEFAULT_METRICS = [
    StdMetric,
    CvMetric,
    InterQuartileRangeMetric,
    RelativeMedianDifferenceRangeMetric,
    ThresholdOutliersCountMetric,
    IqrOutliersCountMetric,
    RegressionSlopeMetric
]
