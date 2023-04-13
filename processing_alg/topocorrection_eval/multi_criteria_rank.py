import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.groupby import SeriesGroupBy
from qgis.core import (
    QgsProcessingParameterMatrix,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingContext,
    QgsProcessingParameterBoolean
)
from tabulate import tabulate

from .eval import EvaluationAlgorithm, MergeStrategy
from .metrics import EvalMetric, EvalContext, DEFAULT_METRICS
from .multi_criteria_eval import MultiCriteriaEvaluationProcessingAlgorithm, GroupResult, DataFrameResult
from ..execution_context import QgisExecutionContext, SerializableQgisExecutionContext
from ..gui.keyed_table_widget import KeyedTableWidgetWrapper
from ..topocorrection import DEFAULT_CORRECTIONS
from ..topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ...computation.gdal_utils import open_img
from ...computation.qgis_utils import init_qgis_env, table_from_matrix_list


class BandMetricsCombiner:
    class Strategy(str, Enum):
        MAX = 'max'
        MIN = 'min'
        MEDIAN = 'median'
        MEAN = 'mean'
        SUM = 'sum'

    DEFAULT_STRATEGIES = {
        Strategy.MAX: lambda values: values.max(),
        Strategy.MIN: lambda values: values.min(),
        Strategy.MEAN: lambda values: values.mean(),
        Strategy.MEDIAN: lambda values: values.median(),
        Strategy.SUM: lambda values: values.sum()
    }

    def __init__(self, combine_strategy: Strategy = None):
        self.combine_strategy = combine_strategy

    def combine(self, scores_per_band: Series) -> Series:
        scores_by_correction = scores_per_band.groupby(level=0)
        return self._combine_single_metric(scores_by_correction)

    def _combine_single_metric(self, values: SeriesGroupBy) -> Series:
        if self.combine_strategy is None:
            raise ValueError()

        return self.DEFAULT_STRATEGIES[self.combine_strategy](values)


class MultiCriteriaRankAlgorithm(EvaluationAlgorithm, MergeStrategy):
    ORIGINAL_IMAGE_KEY = '__orig__'

    def __init__(
            self,
            ctx: QgisExecutionContext,
            metrics: list[EvalMetric],
            corrections: list[TopoCorrectionAlgorithm],
            metrics_combine_strategy: BandMetricsCombiner.Strategy = BandMetricsCombiner.Strategy.SUM,
            group_ids_path=None):
        super().__init__(ctx, self, group_ids_path)
        self.metrics_dict: dict[str, EvalMetric] = {metric.id(): metric for metric in metrics}
        self.corrections = corrections
        self.correction_results = dict()
        self.metrics_combiner = BandMetricsCombiner(metrics_combine_strategy)

    def evaluate(self):
        if self.ctx.run_parallel:
            self._perform_topo_corrections_parallel()
        else:
            self._perform_topo_corrections()
        return super().evaluate()

    def _evaluate_group(self, group_idx):
        group_df = self._compute_metrics_df(group_idx)

        metrics_per_correction_band_df, normalized_metrics = self.merge_strategy.merge(group_df.copy(), group_idx)
        scores_per_correction = self.metrics_combiner.combine(metrics_per_correction_band_df)

        scores_per_correction_df = scores_per_correction.to_frame(name='Score')
        scores_per_correction_df.sort_values(by='Score', ascending=False, inplace=True)

        return [GroupResult(group_idx, {
            'Scores': DataFrameResult(scores_per_correction_df, ['Correction']),
            'Metrics': DataFrameResult(group_df, ['Correction', 'Band']),
            'Normalized metrics': DataFrameResult(normalized_metrics, ['Correction', 'Band']),
            'Per band metrics': DataFrameResult(metrics_per_correction_band_df.to_frame(name='Score'), ['Correction', 'Band'])
        })]

    def _compute_metrics_df(self, group_idx):
        corrected_metrics_dict: dict[str, list[list[float]]] = defaultdict(list)
        corrected_ds_dict = {correction: open_img(result_path)
                             for correction, result_path in self.correction_results.items()}

        luminance_bytes = self._get_masked_bytes(self.ctx.luminance_bytes.ravel(), group_idx)

        orig_stats: list[dict[str, Any]] = []
        for band_idx in range(self.input_ds.RasterCount):
            orig_band = self._get_masked_band(self.input_ds, band_idx, group_idx)
            stats = self._compute_stats(orig_band.bytes)
            orig_metrics = self._evaluate_metrics(
                EvalContext(orig_band, stats, luminance_bytes)
            )
            orig_stats.append(stats)
            corrected_metrics_dict[self.ORIGINAL_IMAGE_KEY].append(orig_metrics)

        for correction_id, corrected_ds in corrected_ds_dict.items():
            for band_idx in range(corrected_ds.RasterCount):
                corrected_band = self._get_masked_band(corrected_ds, band_idx, group_idx)
                band_result = self._evaluate_metrics(
                    EvalContext(corrected_band, orig_stats[band_idx], luminance_bytes)
                )
                corrected_metrics_dict[correction_id].append(band_result)

        del luminance_bytes

        band_dfs = {correction_name: pd.DataFrame(metrics, columns=self.metrics_dict.keys()) for
                    correction_name, metrics in
                    corrected_metrics_dict.items()}

        group_df = pd.concat(band_dfs)
        group_df.index.rename(['Correction', 'Band'], inplace=True)
        return group_df

    def _compute_stats(self, data):
        return {
            'min': np.min(data),
            'max': np.max(data)
        }

    def _evaluate_metrics(self, ctx: EvalContext) -> list[float]:
        return [metric.evaluate(ctx.current_band.bytes, ctx) for metric in self.metrics_dict.values()]

    def merge(self, metrics: DataFrame, group_idx):
        orig_metrics = metrics.xs(self.ORIGINAL_IMAGE_KEY)

        for metric_id, metric in self.metrics_dict.items():
            metrics[metric_id] = metric.combine(orig_metrics[metric_id], metrics[metric_id])

        weights = [metric.weight for metric in self.metrics_dict.values()]
        correction_metrics = metrics.drop(self.ORIGINAL_IMAGE_KEY)

        normalized_metrics: DataFrame = self._normalize(correction_metrics, orig_metrics)
        return (normalized_metrics * weights).sum(1), normalized_metrics

    def _normalize(self, group_result: DataFrame, orig_metrics: DataFrame) -> DataFrame:
        good_results = group_result.where(group_result.ge(orig_metrics, level=1))
        norm_good_results = self._normalize_single(good_results)

        bad_results = group_result.where(group_result.lt(orig_metrics, level=1))
        norm_bad_results = self._normalize_single(bad_results) - 1

        norm_good_results.fillna(norm_bad_results, inplace=True)
        return norm_good_results.replace([np.inf, np.nan], 1)

    def _normalize_single(self, metrics: DataFrame):
        metrics_min, metrics_max = metrics.min(level=1), metrics.max(level=1)
        return metrics.subtract(metrics_min, level=1).divide(metrics_max - metrics_min, level=1)

    def _perform_topo_corrections_parallel(self):
        correction_ctx = SerializableQgisExecutionContext.from_ctx(self.ctx)

        futures = dict()
        with ProcessPoolExecutor() as executor:
            for correction in self.corrections:
                corrected_image_path = os.path.join(self.ctx.output_file_path, f"{correction.get_name()}.tif")
                correction_future = executor.submit(topo_correction_entrypoint, correction_ctx, correction,
                                                    corrected_image_path)
                self.ctx.log(f"Path for {correction.get_name()} is {corrected_image_path}")
                futures[correction.get_name()] = correction_future

            for correction_name, future in futures.items():
                self.correction_results[correction_name] = future.result()
                self.ctx.log(f"get res for {correction_name}")

    def _perform_topo_corrections(self):
        correction_ctx = copy(self.ctx)
        correction_ctx.need_load = False

        for correction in self.corrections:
            self._perform_topo_correction(correction, correction_ctx)

    def _perform_topo_correction(self, correction, ctx):
        corrected_image_path = os.path.join(self.ctx.output_file_path, f"{correction.get_name()}.tif")
        ctx.output_file_path = corrected_image_path
        correction.process(ctx)
        self.correction_results[correction.get_name()] = corrected_image_path


def topo_correction_entrypoint(ctx, correction, corrected_image_path):
    _ = init_qgis_env(ctx.output_file_path)
    ctx.output_file_path = corrected_image_path
    correction.process(ctx)
    return corrected_image_path


class MultiCriteriaRankProcessingAlgorithm(MultiCriteriaEvaluationProcessingAlgorithm):
    def __init__(self):
        super().__init__()
        self.correction_classes = DEFAULT_CORRECTIONS
        self.metric_classes = {MetricClass.name(): MetricClass for MetricClass in DEFAULT_METRICS}

    def group(self):
        return self.tr('Topographic correction ranking')

    def groupId(self):
        return 'topocorrection_rank'

    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)

        self.addParameter(
            QgsProcessingParameterEnum(
                'TOPO_CORRECTION_ALGORITHMS',
                self.tr('Topographic correction algorithms to evaluate'),
                options=[c.get_name() for c in self.correction_classes],
                allowMultiple=True,
                defaultValue=[idx for idx, _ in enumerate(self.correction_classes)]
            )
        )

        metric_merge_strategy_param = QgsProcessingParameterEnum(
            'METRIC_MERGE_STRATEGY',
            self.tr('Strategy for band scores merging'),
            options=[s for s in BandMetricsCombiner.Strategy],
            allowMultiple=False,
            defaultValue='sum',
            usesStaticStrings=True
        )
        self._additional_param(metric_merge_strategy_param)

        parallel_param = QgsProcessingParameterBoolean(
            'RUN_PARALLEL',
            self.tr('Run processing in parallel'),
            defaultValue=False,
            optional=True
        )
        self._additional_param(parallel_param)

        task_timeout_param = QgsProcessingParameterNumber(
            'TASK_TIMEOUT',
            self.tr('Parallel task timeout in ms'),
            defaultValue=10000,
            type=QgsProcessingParameterNumber.Integer
        )
        self._additional_param(task_timeout_param)

    def _metrics_param(self):
        def _default_metrics_gen():
            for metric in self.metric_classes:
                yield metric
                yield 1.0

        metrics = QgsProcessingParameterMatrix(
            'METRICS',
            self.tr('Metrics'),
            numberRows=len(self.metric_classes),
            hasFixedNumberRows=True,
            headers=['Metric', 'Weight'],
            defaultValue=list(_default_metrics_gen())
        )
        metrics.setMetadata({
            'widget_wrapper': {
                'class': KeyedTableWidgetWrapper
            }
        })
        return metrics

    def createInstance(self):
        return MultiCriteriaRankProcessingAlgorithm()

    def name(self):
        return 'multi_criteria_rank'

    def displayName(self):
        return self.tr('Rank TOC algorithms by multi-criteria score')

    def shortHelpString(self):
        return self.tr("Rank TOC algorithms by multi-criteria score based on statistical metrics. "
                       "Current implementation contains following metrics: \n"
                       + '\n'.join([f'<b>{metric.id()}</b>: {metric.name()}' for metric in self.metric_classes.values()])
                       + "\n<b>Note:</b> the illumination model of the input raster image is calculated automatically, "
                       "based on the provided DEM layer. Currently, the input raster image and the DEM must have "
                       "the same CRS, extent and spatial resolution.")

    def _get_scores_per_groups(self, ctx: QgisExecutionContext, group_ids_path: str):
        correction_ids = self.parameterAsEnums(ctx.qgis_params, 'TOPO_CORRECTION_ALGORITHMS', ctx.qgis_context)
        metric_merge_strategy = self.parameterAsEnumString(ctx.qgis_params, 'METRIC_MERGE_STRATEGY', ctx.qgis_context)

        metrics_list = self.parameterAsMatrix(ctx.qgis_params, 'METRICS', ctx.qgis_context)
        metrics_dict = table_from_matrix_list(metrics_list)

        metrics = []
        for metric_id, weight in metrics_dict.items():
            metric = self.metric_classes[metric_id](weight=float(weight[0]))
            metrics.append(metric)

        algorithm = MultiCriteriaRankAlgorithm(
            ctx,
            metrics=metrics,
            corrections=[self.correction_classes[idx]() for idx in correction_ids],
            metrics_combine_strategy=BandMetricsCombiner.Strategy(metric_merge_strategy),
            group_ids_path=group_ids_path
        )

        return algorithm.evaluate()

    def _ctx_additional_kw_args(
            self,
            parameters: Dict[str, Any],
            context: QgsProcessingContext) -> Dict[str, Any]:

        return {
            'sza_degrees': self.parameterAsDouble(parameters, 'SZA', context),
            'solar_azimuth_degrees': self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context),
            'run_parallel': self.parameterAsBoolean(parameters, 'RUN_PARALLEL', context),
            'task_timeout': self.parameterAsInt(parameters, 'TASK_TIMEOUT', context)
        }

    def _log_result(self, ctx: QgisExecutionContext, group_result: GroupResult):
        ctx.log(f"------------------ Results for group-{group_result.group_idx}:")
        formatted_table = tabulate(group_result.data_frames['Scores'].df, headers='keys',
                                   tablefmt='simple_outline')
        ctx.log(formatted_table)
