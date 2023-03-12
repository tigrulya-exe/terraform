import os
from typing import Dict, Any

from qgis.PyQt.QtCore import QCoreApplication
from qgis._core import QgsProcessingParameterFileDestination, QgsProcessingParameterBoolean
from qgis.core import (QgsProcessingContext,
                       QgsProcessingFeedback,
                       QgsProcessingParameterRasterLayer)

from ...computation.qgis_utils import add_layer_to_project
from ..execution_context import QgisExecutionContext
from ..terraform_algorithm import TerraformProcessingAlgorithm


class TopocorrectionEvaluationAlgorithm(TerraformProcessingAlgorithm):
    def __init__(self):
        super().__init__()

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

        self.addParameter(
            QgsProcessingParameterFileDestination(
                'IMG_PLOT_OUT_PATH',
                self.tr('Output path of rose diagram'),
                fileFilter="(*.png *.svg)"
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                'OPEN_OUT_FILE',
                self.tr('Open output file after algorithm finish'),
                defaultValue=True
            )
        )

    def processAlgorithm(
            self,
            parameters: Dict[str, Any],
            context: QgsProcessingContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        input_layer = self.parameterAsRasterLayer(parameters, 'INPUT', context)
        dem_layer = self.parameterAsRasterLayer(parameters, 'DEM', context)
        output_file_path = self._get_valid_outpath(parameters, context)

        execution_ctx = QgisExecutionContext(
            context,
            feedback,
            parameters,
            input_layer,
            dem_layer,
            output_file_path
        )

        result = self.processAlgorithmInternal(parameters, execution_ctx, feedback)

        if self.parameterAsBoolean(parameters, 'OPEN_OUT_FILE', context) and output_file_path.endswith('png'):
            add_layer_to_project(context, output_file_path, f"Result_{self.name()}")

        return result

    def processAlgorithmInternal(
            self,
            parameters: Dict[str, Any],
            context: QgisExecutionContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        pass

    def _get_valid_outpath(self, parameters: Dict[str, Any], context: QgsProcessingContext) -> str:
        output_file_path = self.parameterAsFileOutput(parameters, 'IMG_PLOT_OUT_PATH', context)

        pre, ext = os.path.splitext(output_file_path)
        return pre + '.png' if ext not in ('.png', '.svg') else output_file_path
