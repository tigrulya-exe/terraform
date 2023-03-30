import functools

import numpy as np


def minmax(values: list):
    min_val = values[0]
    max_val = values[0]
    for i in values:
        if i < min_val:
            min_val = i
        if i > max_val:
            max_val = i

    return min_val, max_val


class EvalMetric:
    def __init__(self, is_reduction=True, weight=1.0):
        self.multiplier = 1 if is_reduction else -1
        self.weight = weight

    def id(self):
        pass

    def name(self):
        pass

    def evaluate(self, values) -> float:
        pass

    def metric(self, values) -> float:
        pass

    def combine(self, left: float, right: float):
        pass

    def eval(self, original_bytes: list, corrected_bytes: list) -> float:
        return self.multiplier * (self.metric(original_bytes) - self.metric(corrected_bytes))

    @staticmethod
    def norm(metrics: list[float]) -> list[float]:
        min_val, max_val = minmax(metrics)
        minmax_diff = max_val - min_val
        return [(metric - min_val) / minmax_diff for metric in metrics]


class StdMetric(EvalMetric):
    def metric(self, values) -> float:
        return np.std(values)


class CvMetric(EvalMetric):
    def metric(self, values: list):
        return np.std(values) / np.mean(values)


# todo add iqr outliers metric https://www.scribbr.com/statistics/outliers/
class InterQuartileRangeMetric(EvalMetric):
    @staticmethod
    @functools.cache
    def get_q1_q3(values):
        return np.percentile(values, [25, 75])

    def metric(self, values: list):
        q1, q3 = InterQuartileRangeMetric.get_q1_q3(values)
        return q3 - q1


class RelativeMedianDifferenceRangeMetric(EvalMetric):
    @functools.cache
    def metric(self, values: list):
        return np.median(values)

    def eval(self, original_bytes: list, corrected_bytes: list) -> float:
        orig_metric = self.metric(original_bytes)
        return abs(self.metric(corrected_bytes) - orig_metric) / orig_metric

# todo do smth with threshold
class ThresholdOutliersCountMetric(EvalMetric):
    def __init__(self, threshold):
        super().__init__(is_reduction=False)
        self.threshold = threshold

    def metric(self, values) -> float:
        return np.count_nonzero(values > self.threshold)


class RegressionMetric(EvalMetric):
    def __init__(self, y_bytes):
        super().__init__()
        self.y_bytes = y_bytes

    def eval(self, original_bytes: list, corrected_bytes: list) -> float:
        pass
