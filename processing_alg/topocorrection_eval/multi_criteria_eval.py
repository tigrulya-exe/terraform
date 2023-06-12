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
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame, ExcelWriter
from qgis.core import (
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterLayer,
    QgsProcessingFeedback,
    QgsProcessingParameterNumber,
    QgsProcessingContext
)

from .eval import EvaluationAlgorithm, MergeStrategy
from .metrics import EvalMetric, EvalContext, DEFAULT_METRICS
from .qgis_algorithm import TopocorrectionEvaluationAlgorithm
from ..execution_context import QgisExecutionContext


@dataclass
class DataFrameResult:
    df: DataFrame
    columns: list[str] = field(default_factory=list)


@dataclass
class GroupResult:
    group_idx: int
    data_frames: dict[str, DataFrameResult]


class MultiCriteriaEvalAlgorithm(EvaluationAlgorithm, MergeStrategy):
    def __init__(
            self,
            ctx: QgisExecutionContext,
            metrics: list[EvalMetric],
            group_ids_path=None):
        super().__init__(ctx, self, group_ids_path)
        self.metrics_dict: dict[str, EvalMetric] = {metric.id(): metric for metric in metrics}
        self.correction_results = dict()

    def _evaluate_raster(self, raster_ds, group_idx):
        self.luminance_bytes = self._get_masked_bytes(self.ctx.luminance_bytes.ravel(), group_idx)
        result = super()._evaluate_raster(raster_ds, group_idx)
        del self.luminance_bytes
        return result

    def _evaluate_band(self, band: EvaluationAlgorithm.BandInfo, group_idx) -> Any:
        stats = self._compute_stats(band.bytes)
        orig_metrics = self._evaluate_metrics(
            EvalContext(band, stats, self.luminance_bytes)
        )
        return orig_metrics

    def merge(self, metrics: list[list[float]], group_idx):
        metrics_df = pd.DataFrame(metrics, columns=self.metrics_dict.keys())
        return [GroupResult(group_idx, {
            'metrics': DataFrameResult(metrics_df, ['Band'])
        })]

    def _compute_stats(self, data):
        return {
            'min': np.min(data),
            'max': np.max(data)
        }

    def _evaluate_metrics(self, ctx: EvalContext) -> list[float]:
        return [metric.evaluate(ctx.current_band.bytes, ctx) for metric in self.metrics_dict.values()]


class MultiCriteriaEvaluationProcessingAlgorithm(TopocorrectionEvaluationAlgorithm):
    def __init__(self):
        super().__init__()
        self.metric_classes = DEFAULT_METRICS

    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)

        self.addParameter(
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Solar zenith angle (in degrees)'),
                defaultValue=0.0,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar azimuth (in degrees)'),
                defaultValue=0.0,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(self._metrics_param())

        classification_map_param = QgsProcessingParameterRasterLayer(
            'CLASSIFICATION_MAP',
            self.tr('Raster layer with classification label ids for input raster'),
            optional=True
        )
        self._additional_param(classification_map_param)

    def _metrics_param(self):
        return QgsProcessingParameterEnum(
            'METRICS',
            self.tr('Metrics'),
            options=[c.name() for c in self.metric_classes],
            allowMultiple=True,
            defaultValue=[idx for idx, _ in enumerate(self.metric_classes)]
        )

    def createInstance(self):
        return MultiCriteriaEvaluationProcessingAlgorithm()

    def name(self):
        return 'multi_criteria_eval'

    def displayName(self):
        return self.tr('Multi-criteria TOC evaluation')

    def shortHelpString(self):
        return self.tr("Multi-criteria TOC evaluation based on statistical metrics. "
                       "Current implementation contains following metrics: \n"
                       + '\n'.join([f'<b>{metric.id()}</b>: {metric.name()}' for metric in self.metric_classes])
                       + "\n<b>Note:</b> the illumination model of the input raster image is calculated automatically, "
                         "based on the provided DEM layer. Currently, the input raster image and the DEM must have "
                         "the same CRS, extent and spatial resolution.")

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

    def _process_internal(
            self,
            parameters: Dict[str, Any],
            ctx: QgisExecutionContext,
            feedback: QgsProcessingFeedback):

        group_ids_layer = self.parameterAsRasterLayer(parameters, 'CLASSIFICATION_MAP', ctx.qgis_context)
        group_ids_path = None if group_ids_layer is None else group_ids_layer.source()

        scores_per_group = self._get_scores_per_groups(ctx, group_ids_path)

        output_dir = self._get_output_dir(qgis_params=parameters, qgis_context=ctx.qgis_context)
        output_files = self._export_results(ctx, scores_per_group, output_dir)
        return output_files

    def _get_scores_per_groups(
            self,
            ctx: QgisExecutionContext,
            group_ids_path: str):

        metric_ids = self.parameterAsEnums(ctx.qgis_params, 'METRICS', ctx.qgis_context)
        metrics = [self.metric_classes[idx]() for idx in metric_ids]

        algorithm = MultiCriteriaEvalAlgorithm(
            ctx,
            metrics=metrics,
            group_ids_path=group_ids_path
        )

        return algorithm.evaluate()

    def _ctx_additional_kw_args(
            self,
            parameters: Dict[str, Any],
            context: QgsProcessingContext) -> Dict[str, Any]:

        return {
            'sza_degrees': self.parameterAsDouble(parameters, 'SZA', context),
            'solar_azimuth_degrees': self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context)
        }

    def _export_results(
            self,
            ctx: QgisExecutionContext,
            scores_per_group: list[GroupResult],
            output_directory_path: str = None):

        output_files = dict()
        for group_result in scores_per_group:
            self._log_result(ctx, group_result)

            if output_directory_path is not None:
                out_path = os.path.join(output_directory_path,
                                        f"{ctx.input_file_name}_ranks_group_{group_result.group_idx}.xlsx")
                self._export_to_excel(out_path, group_result)
                output_files[f'OUT_{group_result.group_idx}'] = out_path
        return output_files

    def _log_result(self, ctx: QgisExecutionContext, group_result: GroupResult):
        pass

    def _export_to_excel(self, output_path: str, group_result: GroupResult):
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

        for sheet_name, df_result in group_result.data_frames.items():
            self._export_excel_sheet(writer, df_result.df, sheet_name, columns=df_result.columns)

        writer.save()

    def _export_excel_sheet(self, writer: ExcelWriter, df: DataFrame, sheet_name: str, columns: list[str]):
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

        col_num = 0
        for column in columns:
            worksheet.write(0, col_num, column, header_format)
            col_num += 1

        worksheet.set_column(0, 0, 20)

        # Write the column headers with the defined format.
        for column in df.columns.values:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            worksheet.write(0, col_num, column, header_format)
            worksheet.set_column(col_num, col_num, column_length)
            col_num += 1
