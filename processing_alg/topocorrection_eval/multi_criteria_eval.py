import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import pandas as pd
from pandas import DataFrame, Series, ExcelWriter
from pandas.core.groupby import SeriesGroupBy
from qgis.core import QgsProcessingParameterFolderDestination, QgsProcessingParameterEnum, \
    QgsProcessingParameterRasterLayer, \
    QgsProcessingFeedback, QgsProcessingParameterNumber, QgsProcessingException, QgsTask, QgsTaskManager
from tabulate import tabulate

from .eval import EvaluationAlgorithm, MergeStrategy
from .metrics import EvalMetric, EvalContext, DEFAULT_METRICS
from .qgis_algorithm import TopocorrectionEvaluationAlgorithm
from ..execution_context import QgisExecutionContext, SerializableCorrectionExecutionContext
from ..topocorrection import DEFAULT_CORRECTIONS
from ..topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ...computation.gdal_utils import open_img
from ...computation.qgis_utils import init_qgis_env


@dataclass
class GroupResult:
    group_idx: int
    score_per_correction: DataFrame
    extended_metrics: DataFrame
    normalized_metrics: DataFrame


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


def topo_correction_entrypoint(ctx, correction, corrected_image_path):
    init_qgis_env(ctx.qgis_path)
    ctx.output_file_path = corrected_image_path
    correction.process(ctx)
    return corrected_image_path


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
        column_names = self.metrics_dict.keys()
        corrected_metrics_dict: dict[str, list[list[float]]] = defaultdict(list)

        corrected_ds_dict = {correction: open_img(result_path)
                             for correction, result_path in self.correction_results.items()}

        for band_idx in range(self.input_ds.RasterCount):
            orig_band = self._get_masked_band(self.input_ds, band_idx, group_idx)
            orig_metrics = self._evaluate_band_unary(orig_band)

            for correction_id, corrected_ds in corrected_ds_dict.items():
                corrected_band = self._get_masked_band(corrected_ds, band_idx, group_idx)
                band_result = self._evaluate_band_binary(
                    EvalContext(corrected_band, orig_band, orig_metrics)
                )
                corrected_metrics_dict[correction_id].append(band_result)

        band_dfs = {correction_name: pd.DataFrame(metrics, columns=column_names) for correction_name, metrics in
                    corrected_metrics_dict.items()}
        group_df = pd.concat(band_dfs)

        metrics_per_correction_band, normalized_metrics = self.merge_strategy.merge(group_df)
        scores_per_correction = self.metrics_combiner.combine(metrics_per_correction_band)

        scores_per_correction_df = scores_per_correction.to_frame(name='Score')

        return [GroupResult(group_idx, scores_per_correction_df, group_df, normalized_metrics)]

    def _evaluate_band_unary(self, band: EvaluationAlgorithm.BandInfo) -> dict[str, float]:
        metrics_results = dict()
        for metric_id, metric in self.metrics_dict.items():
            metrics_results[metric.id()] = metric.unary(band.bytes)

        return metrics_results

    def _evaluate_band_binary(self, ctx: EvalContext) -> list[float]:
        return [metric.binary(ctx.current_band.bytes, ctx) for metric in self.metrics_dict.values()]

    def merge(self, metrics: DataFrame):
        weights = [metric.weight for metric in self.metrics_dict.values()]
        normalized_metrics: DataFrame = self._norm(metrics)
        return (normalized_metrics * weights).sum(1), normalized_metrics

    def _norm(self, group_result: DataFrame) -> DataFrame:
        metrics_min = group_result.min()
        normalized_df = (group_result - metrics_min) / (group_result.max() - metrics_min)

        for metric_id, metric in self.metrics_dict.items():
            if metric.is_reduction:
                normalized_df[metric_id] = 1 - normalized_df[metric_id]

        return normalized_df

    # todo parallelize
    def _perform_topo_corrections(self):
        correction_ctx = SerializableCorrectionExecutionContext.from_ctx(self.ctx)

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
    def __init__(self):
        super().__init__()
        self.correction_classess = DEFAULT_CORRECTIONS
        self.metric_classes = DEFAULT_METRICS

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

        self.addParameter(
            QgsProcessingParameterEnum(
                'TOPO_CORRECTION_ALGORITHMS',
                self.tr('Topographic correction algorithms to evaluate'),
                options=[c.get_name() for c in self.correction_classess],
                allowMultiple=True,
                defaultValue=[idx for idx, _ in enumerate(self.correction_classess)]
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                'METRICS',
                self.tr('Metrics'),
                options=[c.name() for c in self.metric_classes],
                allowMultiple=True,
                defaultValue=[idx for idx, _ in enumerate(self.metric_classes)]
            )
        )

        metric_merge_strategy_param = QgsProcessingParameterEnum(
            'METRIC_MERGE_STRATEGY',
            self.tr('Strategy for band scores merging'),
            options=[s for s in BandMetricsCombiner.Strategy],
            allowMultiple=False,
            defaultValue='max',
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

    def add_output_param(self):
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                'OUTPUT_DIR',
                self.tr('Output directory'),
                optional=True,
                defaultValue=None
            )
        )

        return 'OUTPUT_DIR'

    def processAlgorithmInternal(
            self,
            parameters: Dict[str, Any],
            ctx: QgisExecutionContext,
            feedback: QgsProcessingFeedback):
        ctx.sza_degrees = self.parameterAsDouble(parameters, 'SZA', ctx.qgis_context)
        ctx.solar_azimuth_degrees = self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', ctx.qgis_context)

        metric_merge_strategy = self.parameterAsEnumString(parameters, 'METRIC_MERGE_STRATEGY', ctx.qgis_context)

        group_ids_layer = self.parameterAsRasterLayer(parameters, 'CLASSIFICATION_MAP', ctx.qgis_context)
        group_ids_path = None if group_ids_layer is None else group_ids_layer.source()

        metric_ids = self.parameterAsEnums(parameters, 'METRICS', ctx.qgis_context)
        correction_ids = self.parameterAsEnums(parameters, 'TOPO_CORRECTION_ALGORITHMS', ctx.qgis_context)

        algorithm = MultiCriteriaEvalAlgorithm(
            ctx,
            [self.metric_classes[idx]() for idx in metric_ids],
            [self.correction_classess[idx]() for idx in correction_ids],
            BandMetricsCombiner.Strategy(metric_merge_strategy),
            group_ids_path
        )

        scores_per_group = algorithm.evaluate()
        output_dir = self._get_output_dir(qgis_params=parameters, qgis_context=ctx.qgis_context)
        self._print_results(ctx, scores_per_group, output_dir)
        # no raster output
        return []

    def _print_results(
            self,
            ctx: QgisExecutionContext,
            scores_per_group: list[GroupResult],
            output_directory_path: str = None):
        for group_result in scores_per_group:
            ctx.log(f"------------------ Results for group-{group_result.group_idx}:")
            group_result.score_per_correction.sort_values(by='Score', ascending=False, inplace=True)
            formatted_table = tabulate(group_result.score_per_correction, headers='keys', tablefmt='simple_outline')
            ctx.log(formatted_table)

            if output_directory_path is not None:
                out_path = os.path.join(output_directory_path, f"group_{group_result.group_idx}.xlsx")
                self._export_to_excel(out_path, group_result)

    def _export_to_excel(self, output_path: str, group_result: GroupResult):
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

        self._export_excel_sheet(writer, group_result.score_per_correction, 'Scores')
        self._export_excel_sheet(writer, group_result.extended_metrics, 'Metrics', column_offset=2)
        self._export_excel_sheet(writer, group_result.normalized_metrics, 'Normalized metrics', column_offset=2)

        writer.save()

    def _export_excel_sheet(self, writer: ExcelWriter, df: DataFrame, sheet_name: str, column_offset=1):
        worksheet = writer.book.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False)

        # Add a header format.
        header_format = writer.book.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1})

        # worksheet.write(0, 1, "Band", header_format)
        worksheet.set_column(0, 0, 20)

        # Write the column headers with the defined format.
        for col_num, column in enumerate(df.columns.values):
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            worksheet.write(0, col_num + column_offset, column, header_format)
            worksheet.set_column(col_num + column_offset, col_num + column_offset, column_length)
