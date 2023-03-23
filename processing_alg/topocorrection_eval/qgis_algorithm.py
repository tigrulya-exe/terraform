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
from ...computation.qgis_utils import set_layers_to_load


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
        # 'INPUT' is the recommended name for the main input
        # parameter.
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
            context,
            feedback,
            parameters,
            input_layer,
            dem_layer,
            output_file_path
        )

        result = self.processAlgorithmInternal(parameters, execution_ctx, feedback)

        self.add_layers_to_project(execution_ctx, result)

        return {
            "OUT": result
        }

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

    def processAlgorithmInternal(
            self,
            parameters: Dict[str, Any],
            context: QgisExecutionContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        pass

    def _get_output_dir(self, qgis_params, qgis_context):
        output_directory = self.parameterAsString(qgis_params, 'OUTPUT_DIR', qgis_context)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        return output_directory
