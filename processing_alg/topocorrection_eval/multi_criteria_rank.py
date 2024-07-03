#!/usr/bin/env python
""" Terraform QGIS plugin.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = 'Tigran Manasyan'
__copyright__ = '(C) 2023 by Tigran Manasyan'
__license__ = "GPLv3"

import os
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.groupby import SeriesGroupBy
from qgis._core import QgsProcessingParameterFolderDestination
from qgis.core import (
    QgsProcessingParameterMatrix,
    QgsProcessingParameterEnum,
    QgsProcessingContext
)
from tabulate import tabulate

from .eval import EvaluationAlgorithm, MergeStrategy
from .metrics import EvalMetric, EvalContext, DEFAULT_METRICS
from .multi_criteria_eval import MultiCriteriaEvaluationProcessingAlgorithm, GroupResult, DataFrameResult
from ..ParallelProcessingParamMixin import ParallelProcessingParamMixin
from ..execution_context import QgisExecutionContext, SerializableQgisExecutionContext
from ..gui.keyed_table_widget import KeyedTableWidgetWrapper
from ..topocorrection import DEFAULT_CORRECTIONS
from ..topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ...util.gdal_utils import open_img
from ...util.qgis_utils import init_qgis_env, table_from_matrix_list


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
            group_ids_path=None,
            correction_results_directory=None):
        super().__init__(ctx, self, group_ids_path)
        self.metrics_dict: dict[str, EvalMetric] = {metric.id(): metric for metric in metrics}
        self.corrections = corrections
        self.correction_results = dict()
        self.metrics_combiner = BandMetricsCombiner(metrics_combine_strategy)
        self.correction_results_directory = correction_results_directory or ctx.tmp_dir

    def evaluate(self):
        self._perform_topo_corrections()
        return super().evaluate()

    def _evaluate_group(self, group_idx):
        group_df = self._compute_metrics_df(group_idx)

        self.ctx.log_info(f"Normalizing metrics.")
        metrics_per_correction_band_df, normalized_metrics = self.merge_strategy.merge(group_df.copy(), group_idx)
        self.ctx.log_info(f"Combining metrics.")
        scores_per_correction = self.metrics_combiner.combine(metrics_per_correction_band_df)

        scores_per_correction_df = scores_per_correction.to_frame(name='Score')
        scores_per_correction_df.sort_values(by='Score', ascending=False, inplace=True)

        return [GroupResult(group_idx, {
            'Scores': DataFrameResult(scores_per_correction_df, ['Correction']),
            'Metrics': DataFrameResult(group_df, ['Correction', 'Band']),
            'Normalized metrics': DataFrameResult(normalized_metrics, ['Correction', 'Band']),
            'Per band metrics': DataFrameResult(metrics_per_correction_band_df.to_frame(name='Score'),
                                                ['Correction', 'Band'])
        })]

    def _compute_metrics_df(self, group_idx):
        corrected_metrics_dict: dict[str, list[list[float]]] = defaultdict(list)
        corrected_ds_dict = {correction: open_img(result_path)
                             for correction, result_path in self.correction_results.items()}

        luminance_bytes = self._get_masked_bytes(self.ctx.luminance_bytes.ravel(), group_idx)

        self.ctx.log_info("Computing metrics for original image.")
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
            self.ctx.log_info(f"Computing metrics for {correction_id}.")
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

        normalized_metrics: DataFrame = self._normalize(metrics, orig_metrics)
        return (normalized_metrics * weights).sum(1), normalized_metrics

    def _normalize(self, group_result: DataFrame, orig_metrics: DataFrame) -> DataFrame:
        good_results = group_result.where(group_result.gt(orig_metrics, level=1))
        norm_good_results = self._normalize_single(good_results, metrics_min=orig_metrics)

        bad_results = group_result.where(group_result.lt(orig_metrics, level=1))
        norm_bad_results = self._normalize_single(bad_results, metrics_max=orig_metrics) - 1

        norm_good_results[group_result.eq(orig_metrics, level=1)] = 0.0

        norm_good_results.fillna(norm_bad_results, inplace=True)
        return norm_good_results.drop(self.ORIGINAL_IMAGE_KEY)

    def _normalize_single(self, metrics: DataFrame, metrics_min=None, metrics_max=None):
        if metrics_min is None:
            metrics_min = metrics.groupby(level=1).min()
        if metrics_max is None:
            metrics_max = metrics.groupby(level=1).max()
        return metrics.subtract(metrics_min, level=1).divide(metrics_max - metrics_min, level=1, )

    def _perform_topo_corrections(self):
        self.ctx.log_info("Starting topographic correction.")
        try:
            if self.ctx.run_parallel:
                self._perform_topo_corrections_parallel()
            else:
                self._perform_topo_corrections_sequential()
        except Exception as exc:
            self.ctx.log_error(f"Error during correction: {traceback.format_exc()}", fatal=True)
            self.ctx.force_cancel(exc)

    def _perform_topo_corrections_parallel(self):
        correction_ctx = SerializableQgisExecutionContext.from_ctx(self.ctx)

        futures = dict()
        with ProcessPoolExecutor(max_workers=self.ctx.worker_count) as executor:
            for correction in self.corrections:
                corrected_image_path = self._build_correction_result_name(correction)
                correction_future = executor.submit(topo_correction_entrypoint, correction_ctx, correction,
                                                    corrected_image_path)
                futures[correction.name()] = correction_future

            for correction_name, future in futures.items():
                self.correction_results[correction_name] = future.result()
                self.ctx.log_info(f"{correction_name} finished.")

    def _perform_topo_corrections_sequential(self):
        correction_ctx = copy(self.ctx)
        correction_ctx.need_load = False

        for correction in self.corrections:
            self._perform_topo_correction(correction, correction_ctx)

    def _build_correction_result_name(self, correction):
        return os.path.join(self.correction_results_directory, f"{correction.name()}.tif")

    def _perform_topo_correction(self, correction, ctx):
        corrected_image_path = self._build_correction_result_name(correction)
        ctx.output_file_path = corrected_image_path
        correction.process(ctx)
        self.correction_results[correction.name()] = corrected_image_path


def topo_correction_entrypoint(ctx, correction, corrected_image_path):
    _ = init_qgis_env(ctx.output_file_path)
    ctx.output_file_path = corrected_image_path
    correction.process(ctx)
    return corrected_image_path


class MultiCriteriaRankProcessingAlgorithm(MultiCriteriaEvaluationProcessingAlgorithm, ParallelProcessingParamMixin):
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
                options=[c.name() for c in self.correction_classes],
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

        params = self.parallel_run_params()
        [self._additional_param(param) for param in params]

    def add_output_param(self):
        main_out_param = super().add_output_param()

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                'CORRECTIONS_OUTPUT_DIR',
                self.tr('Output directory for correction results'),
                optional=True,
                defaultValue=None
            )
        )

        return main_out_param

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
                       + '\n'.join(
            [f'<b>{metric.id()}</b>: {metric.name()}' for metric in self.metric_classes.values()])
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

        correction_results_directory = self._get_output_dir(
            ctx.qgis_params, ctx.qgis_context, param_name='CORRECTIONS_OUTPUT_DIR')

        if correction_results_directory is None:
            raise ValueError("Output directory should not be null")

        algorithm = MultiCriteriaRankAlgorithm(
            ctx,
            metrics=metrics,
            corrections=[self.correction_classes[idx]() for idx in correction_ids],
            metrics_combine_strategy=BandMetricsCombiner.Strategy(metric_merge_strategy),
            group_ids_path=group_ids_path,
            correction_results_directory=correction_results_directory
        )

        return algorithm.evaluate()

    def _ctx_additional_kw_args(
            self,
            parameters: Dict[str, Any],
            context: QgsProcessingContext) -> Dict[str, Any]:

        return {
            'sza_degrees': self.parameterAsDouble(parameters, 'SZA', context),
            'solar_azimuth_degrees': self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context),
            'run_parallel': self.get_run_parallel_param(parameters, context),
            'task_timeout': self.get_parallel_timeout_param(parameters, context),
            'worker_count': self.get_worker_count_param(parameters, context)
        }

    def _log_result(self, ctx: QgisExecutionContext, group_result: GroupResult):
        ctx.log_info(f"------------------ Results for group-{group_result.group_idx}:")
        formatted_table = tabulate(group_result.data_frames['Scores'].df, headers='keys',
                                   tablefmt='simple_outline')
        ctx.log_info(formatted_table)
        ctx.log_info(f"You can find results in the following directory: {ctx.output_file_path}")
