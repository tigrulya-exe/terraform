import os
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum
from statistics import mean, median
from typing import Any, Dict

import pandas as pd
from qgis.core import QgsProcessingParameterEnum, QgsProcessingParameterRasterLayer, \
    QgsProcessingFeedback, QgsProcessingParameterNumber, QgsProcessingException, QgsTask, QgsTaskManager
from tabulate import tabulate

from .eval import EvaluationAlgorithm, MergeStrategy
from .metrics import StdMetric, CvMetric, InterQuartileRangeMetric, RelativeMedianDifferenceRangeMetric, EvalMetric, \
    IqrOutliersCountMetric, EvalContext, RegressionSlopeMetric, ThresholdOutliersCountMetric
from .qgis_algorithm import TopocorrectionEvaluationAlgorithm
from ..execution_context import QgisExecutionContext
from ..topocorrection import DEFAULT_CORRECTIONS
from ..topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ...computation.gdal_utils import open_img


@dataclass
class BandResult:
    metrics: dict[str, float]

    def __str__(self) -> str:
        return f"{self.metrics}\n"


@dataclass
class GroupResult:
    group_idx: int
    corrected_results: dict[str, list[BandResult]]

    def corrected_metrics(self, band_idx, metric_id) -> list[float]:
        return [res[band_idx].metrics[metric_id] for res in self.corrected_results.values()]


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
        corrected_metrics_dict: dict[str, list[BandResult]] = defaultdict(list)

        corrected_ds_dict = {correction: open_img(result_path)
                             for correction, result_path in self.correction_results.items()}

        for band_idx in range(self.input_ds.RasterCount):
            orig_band = self._get_masked_band(self.input_ds, band_idx, group_idx)
            orig_metrics = self._evaluate_band_unary(orig_band).metrics

            for correction_id, corrected_ds in corrected_ds_dict.items():
                corrected_band = self._get_masked_band(corrected_ds, band_idx, group_idx)
                band_result = self._evaluate_band_binary(
                    EvalContext(corrected_band, orig_band, orig_metrics)
                )
                corrected_metrics_dict[correction_id].append(band_result)

        group_result = GroupResult(group_idx, corrected_metrics_dict)

        # self.ctx.log(str(group_result))
        scores_per_band = self.merge_strategy.merge(group_result)

        return [self.metrics_combiner.combine(scores_per_band)]

    def merge(self, results: GroupResult):
        # list of correction_name -> overall score (higher better) for each band
        scores_per_band: list[dict[str, float]] = []
        for band_idx in range(self.input_ds.RasterCount):
            normalized_metrics: dict[str, list[float]] = self._norm(results, band_idx)

            band_scores_per_correction = dict()
            for correction_idx, correction_name in enumerate(results.corrected_results.keys()):
                band_score = 0.0
                for metric_id, normed_values in normalized_metrics.items():
                    band_score += self.metrics_dict[metric_id].weight * normed_values[correction_idx]

                band_scores_per_correction[correction_name] = band_score

            scores_per_band.append(band_scores_per_correction)
        return scores_per_band

    def _evaluate_band_unary(self, band: EvaluationAlgorithm.BandInfo) -> BandResult:
        metrics_results = dict()
        for metric_id, metric in self.metrics_dict.items():
            metrics_results[metric.id()] = metric.unary(band.bytes)

        return BandResult(metrics_results)

    def _evaluate_band_binary(self, ctx: EvalContext) -> BandResult:
        metrics_results = dict()
        for metric_id, metric in self.metrics_dict.items():
            metrics_results[metric.id()] = metric.binary(ctx.current_band.bytes, ctx)

        return BandResult(metrics_results)

    def _get_masked_band(self, ds, band_idx, group_idx) -> EvaluationAlgorithm.BandInfo:
        orig_band = ds.GetRasterBand(band_idx + 1)
        orig_band_bytes = orig_band.ReadAsArray().ravel()[self.groups_map == group_idx]
        return self.BandInfo(orig_band, orig_band_bytes, band_idx)

    def _norm(self, group_result: GroupResult, band_idx: int) -> dict[str, list[float]]:
        norms: dict[str, list[float]] = dict()
        for metric_id, metric in self.metrics_dict.items():
            corrected_metrics = group_result.corrected_metrics(band_idx, metric_id)
            norms[metric_id] = metric.norm(corrected_metrics)
        return norms

    # todo parallelize
    def _perform_topo_corrections(self):
        correction_ctx = copy(self.ctx)
        for correction in self.corrections:
            self._perform_topo_correction(correction, correction_ctx)

    def _perform_topo_corrections_parallel(self):
        task_manager = QgsTaskManager()

        def task_wrapper(task, _correction, _ctx):
            self._perform_topo_correction(_correction, _ctx)

        _ = self.ctx.luminance_path

        for correction in self.corrections:
            try:
                correction_ctx = copy(self.ctx)
                task = QgsTask.fromFunction(f'Task for correction {correction}', task_wrapper,
                                            _correction=correction, _ctx=correction_ctx)
                task_manager.addTask(task)
            except QgsProcessingException as exc:
                task_manager.cancelAll()
                raise RuntimeError(f"Error during performing topocorrection: {exc}")

            if self.ctx.is_canceled():
                task_manager.cancelAll()
                return None

        for task in task_manager.tasks():
            if not task.waitForFinished(10000):
                raise RuntimeError(f"Timeout exception for task {task.description()}")
            self.ctx.log(f"Task {task.description()} finished")
            if self.ctx.is_canceled():
                task_manager.cancelAll()
                return None

    def _perform_topo_correction(self, correction, ctx):
        corrected_image_path = os.path.join(self.ctx.output_file_path, f"{correction.get_name()}.tif")
        ctx.output_file_path = corrected_image_path
        correction.process(ctx)
        self.correction_results[correction.get_name()] = corrected_image_path


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
            RelativeMedianDifferenceRangeMetric(),
            IqrOutliersCountMetric(),
            ThresholdOutliersCountMetric(),
            RegressionSlopeMetric()
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
            df = pd.DataFrame(scores_dict.items(), columns=["Correction", "Score"])
            df.sort_values('Score', ascending=False, inplace=True, ignore_index=True)
            ctx.log(tabulate(df, headers='keys', tablefmt='simple_outline'))
