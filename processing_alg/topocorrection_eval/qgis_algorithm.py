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
from typing import Dict, Any

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingContext,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterBoolean,
                       QgsProcessingFeedback,
                       QgsProcessingParameterRasterLayer)

from ..execution_context import QgisExecutionContext
from ..terraform_algorithm import TerraformProcessingAlgorithm
from ...util.qgis_utils import get_project_tmp_dir, set_layers_to_load


class TopocorrectionEvaluationAlgorithm(TerraformProcessingAlgorithm):
    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr('Topographic correction evaluation')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs
        to.
        """
        return 'topocorrection_eval'

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and outputs of the algorithm.
        """

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                'INPUT',
                self.tr('Input raster layer')
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                'DEM',
                self.tr('Input DEM layer')
            )
        )

        self.add_output_param()

    def processAlgorithm(
            self,
            parameters: Dict[str, Any],
            context: QgsProcessingContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        input_layer = self.parameterAsRasterLayer(parameters, 'INPUT', context)
        dem_layer = self.parameterAsRasterLayer(parameters, 'DEM', context)
        output_file_path = self._get_output_dir(parameters, context)

        execution_ctx = QgisExecutionContext(
            qgis_context=context,
            qgis_feedback=feedback,
            qgis_params=parameters,
            input_layer=input_layer,
            dem_layer=dem_layer,
            output_file_path=output_file_path,
            tmp_dir=get_project_tmp_dir(),
            **self._ctx_additional_kw_args(parameters, context)
        )

        result = self._process_internal(parameters, execution_ctx, feedback)

        self.add_layers_to_project(execution_ctx, result)

        return {"OUTPUT": result}

    def add_output_param(self):
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                'OUTPUT_DIR',
                self.tr('Output directory'),
                createByDefault=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                'OPEN_OUT_FILE',
                self.tr('Open generated files in the output directory after running algorithm'),
                defaultValue=True
            )
        )

        return 'OUTPUT_DIR'

    def add_layers_to_project(self, ctx: QgisExecutionContext, results):
        need_open = self.parameterAsBoolean(ctx.qgis_params, 'OPEN_OUT_FILE', ctx.qgis_context)
        if need_open:
            set_layers_to_load(ctx.qgis_context, results)

    def _process_internal(
            self,
            parameters: Dict[str, Any],
            context: QgisExecutionContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        pass

    def _ctx_additional_kw_args(
            self,
            parameters: Dict[str, Any],
            context: QgsProcessingContext) -> Dict[str, Any]:
        return dict()

    def _get_output_dir(self, qgis_params, qgis_context, param_name='OUTPUT_DIR'):
        output_directory = self.parameterAsString(qgis_params, param_name, qgis_context)
        if output_directory == "" or output_directory is None:
            return None

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        return output_directory
