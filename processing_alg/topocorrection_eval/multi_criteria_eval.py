from dataclasses import dataclass
from enum import Enum
from statistics import mean, median
from typing import Any, Dict

from qgis.core import QgsProcessingParameterEnum, QgsProcessingParameterRasterLayer, \
    QgsProcessingFeedback, QgsProcessingParameterNumber

from .eval import EvaluationAlgorithm, MergeStrategy
from .metrics import StdMetric, CvMetric, InterQuartileRangeMetric, RelativeMedianDifferenceRangeMetric, EvalMetric
from .qgis_algorithm import TopocorrectionEvaluationAlgorithm
from ..execution_context import QgisExecutionContext
from ..topocorrection import DEFAULT_CORRECTIONS
from ..topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ...computation.gdal_utils import open_img


@dataclass
class BandResult:
    metrics: dict[str, float]


@dataclass
class RasterResult:
    metrics: list[BandResult]


@dataclass
class GroupResult:
    group_idx: int
    original_result: RasterResult
    corrected_results: dict[str, RasterResult]

    def corrected_metrics(self, band_idx, metric_id) -> list[float]:
        return [res.metrics[band_idx].metrics[metric_id] for res in self.corrected_results.values()]


class BandMetricsCombiner:
    class Strategy(str, Enum):
        MAX = 'max'
        MIN = 'min'
        MEDIAN = 'median'
        MEAN = 'mean'
        SUM = 'sum'
        NORM_SUM = 'normalised_sum'

    DEFAULT_STRATEGIES = {
        Strategy.MAX: lambda values: max(values),
        Strategy.MIN: lambda values: min(values),
        Strategy.MEAN: lambda values: mean(values),
        Strategy.MEDIAN: lambda values: median(values),
        Strategy.SUM: lambda values: sum(values),
        Strategy.NORM_SUM: lambda values: sum(values) / len(values),
    }

    def __init__(self, combine_strategy: Strategy = None):
        self.combine_strategy = combine_strategy

    def combine(self, scores_per_band: list[dict[str, float]]) -> dict[str, float]:
        score_per_metric = dict()
        # todo add validation
        metric_ids = scores_per_band[0].keys()

        for metric_id in metric_ids:
            metric_vals = [scores[metric_id] for scores in scores_per_band]
            score_per_metric[metric_id] = self._combine_single_metric(metric_id, metric_vals)

        return score_per_metric

    def _combine_single_metric(self, metric_id: str, values: list[float]) -> float:
        if self.combine_strategy is None:
            raise ValueError()

        return self.DEFAULT_STRATEGIES[self.combine_strategy](values)


class MultiCriteriaEvalAlgorithm(EvaluationAlgorithm, MergeStrategy):
    def __init__(
            self,
            ctx: QgisExecutionContext,
            metrics: list[EvalMetric],
            corrections: list[TopoCorrectionAlgorithm],
            metrics_combine_strategy: BandMetricsCombiner.Strategy = BandMetricsCombiner.Strategy.MAX,
            group_ids_path=None):
        super().__init__(ctx, self, group_ids_path)
        self.metrics_dict: dict[str, EvalMetric] = {metric.id(): metric for metric in metrics}
        self.corrections = corrections
        self.correction_results = dict()
        self.metrics_combiner = BandMetricsCombiner(metrics_combine_strategy)

    def evaluate(self):
        self._perform_topo_corrections()
        return super().evaluate()

    def _evaluate_group(self, group_idx):
        corrected_metrics_dict = dict()

        self.corrected_handles = [open_img(path) for path in self.correction_results]

        orig_img_result = self._evaluate_raster(self.input_ds, group_idx)

        for correction_name, corrected_path in self.correction_results.items():
            corrected_ds = open_img(corrected_path)
            corrected_metrics = self._evaluate_raster(corrected_ds, group_idx)
            corrected_metrics_dict[correction_name] = corrected_metrics

        group_result = GroupResult(group_idx, orig_img_result, corrected_metrics_dict)
        scores_per_band = self.merge_strategy.merge(group_result)

        return [self.metrics_combiner.combine(scores_per_band)]

    def _evaluate_raster(self, raster_ds, group_idx):
        return RasterResult(
            super()._evaluate_raster(raster_ds, group_idx)
        )

    def _evaluate_band(self, band: EvaluationAlgorithm.BandInfo, group_idx) -> BandResult:
        metrics_results = dict()
        for metric_id, metric in self.metrics_dict:
            metrics_results[metric.id()] = metric.evaluate(band.band_bytes)

        return BandResult(metrics_results)

    def merge(self, results: GroupResult):
        # list of correction_name -> overall score (higher better)
        scores_per_band: list[dict[str, float]] = []
        for band_idx, orig_band_result in enumerate(results.original_result.metrics):
            band_metrics_dict = orig_band_result.metrics
            for correction_name, corrected_result in results.corrected_results.items():
                corrected_band_metrics_dict = corrected_result.metrics[band_idx].metrics
                combined_metrics = self._combine_metrics(band_metrics_dict, corrected_band_metrics_dict)
                corrected_result.metrics[band_idx].metrics = combined_metrics

            normalized_metrics: dict[str, list[float]] = self._norm(results, band_idx)

            band_merge_results = dict()
            for correction_idx, correction_name in enumerate(results.corrected_results.keys()):
                correction_band_result = 0.0
                for metric_id, normed_values in normalized_metrics.items():
                    correction_band_result += self.metrics_dict[metric_id].multiplier * normed_values[correction_idx]

                band_merge_results[correction_name] = correction_band_result

            scores_per_band.append(band_merge_results)
        return scores_per_band

    def _norm(self, group_result: GroupResult, band_idx: int) -> dict[str, list[float]]:
        norms: dict[str, list[float]] = dict()
        for metric_id in self.metrics_dict.keys():
            norms[metric_id] = group_result.corrected_metrics(band_idx, metric_id)
        return norms

    def _combine_metrics(
            self,
            original_band_metrics: dict[str, float],
            corrected_band_metrics: dict[str, float]) -> dict[str, float]:
        combined_metrics = dict()
        for metric_id, orig_metric_val in original_band_metrics.items():
            corrected_metric_val = corrected_band_metrics[metric_id]
            combined_metric = self.metrics_dict[metric_id].combine(orig_metric_val, corrected_metric_val)
            combined_metrics[metric_id] = combined_metric

        return combined_metrics

    def _perform_topo_corrections(self):
        for correction in self.corrections:
            dict_result = correction.process(self.ctx)
            self.correction_results[correction.get_name()] = dict_result['OUT']


class MultiCriteriaEvaluationProcessingAlgorithm(TopocorrectionEvaluationAlgorithm):
    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)

        self.addParameter(
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Solar zenith angle (in degrees)'),
                defaultValue=57.2478878065826,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar azimuth (in degrees)'),
                defaultValue=177.744663052425,
                type=QgsProcessingParameterNumber.Double
            )
        )

        metric_merge_strategy_param = QgsProcessingParameterEnum(
            'METRIC_MERGE_STRATEGY',
            self.tr('Strategy for band scores merging'),
            options=[s for s in BandMetricsCombiner.Strategy],
            allowMultiple=False,
            defaultValue='normalised_sum',
            usesStaticStrings=True
        )
        self._additional_param(metric_merge_strategy_param)

        classification_map_param = QgsProcessingParameterRasterLayer(
            'CLASSIFICATION_MAP',
            self.tr('Raster layer with classification label ids for input raster'),
            optional=True
        )
        self._additional_param(classification_map_param)

    def createInstance(self):
        return MultiCriteriaEvaluationProcessingAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'multi_criteria_eval'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Multi-criteria evaluation')

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('TODO')

    def processAlgorithmInternal(
            self,
            parameters: Dict[str, Any],
            ctx: QgisExecutionContext,
            feedback: QgsProcessingFeedback):
        ctx.sza_degrees = self.parameterAsDouble(parameters, 'SZA', ctx.qgis_context)
        ctx.solar_azimuth_degrees = self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', ctx.qgis_context)

        metric_merge_strategy = self.parameterAsEnumString(parameters, 'METRIC_MERGE_STRATEGY', ctx.qgis_context)
        metrics = [
            StdMetric(),
            CvMetric(),
            InterQuartileRangeMetric(),
            RelativeMedianDifferenceRangeMetric()
        ]

        corrections = [CorrectionClass() for CorrectionClass in DEFAULT_CORRECTIONS]

        group_ids_layer = self.parameterAsRasterLayer(parameters, 'CLASSIFICATION_MAP', ctx.qgis_context)
        group_ids_path = None if group_ids_layer is None else group_ids_layer.source()

        algorithm = MultiCriteriaEvalAlgorithm(
            ctx,
            metrics,
            corrections,
            BandMetricsCombiner.Strategy(metric_merge_strategy),
            group_ids_path
        )

        scores_per_group = algorithm.evaluate()
        self._print_results(ctx, scores_per_group)
        # no raster output
        return []

    # todo add group num in results
    def _print_results(self, ctx: QgisExecutionContext, scores_per_group: list[dict[str, float]]):
        for idx, scores_dict in enumerate(scores_per_group):
            ctx.log(f"------------------ Results for group-{idx}:")
            for correction_name, score in scores_dict.items():
                ctx.log(f"{correction_name}'s score: {score}")
