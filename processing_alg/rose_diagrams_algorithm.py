from typing import Dict, Any

from qgis.PyQt.QtCore import QCoreApplication
from qgis._core import QgsProcessingParameterFileDestination
from qgis.core import (QgsProcessingContext,
                       QgsProcessingFeedback,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterString)

import computation.statistics
from processing_alg.terraform_processing_algorithm import TerraformProcessingAlgorithm


class RoseDiagramsEvaluationAlgorithm(TerraformProcessingAlgorithm):
    def __init__(self):
        super().__init__()

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        # Must return a new copy of your algorithm.
        return RoseDiagramsEvaluationAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'terraform_rosediag_eval'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Rose diagram topocorrection algorithm evaluation')

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Builds rose diagrams to evaluate topocorrection algorithm correctness.')

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
                fileFilter="*.png"
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

        slope_path = self.build_slope_layer(feedback, context, dem_layer, in_radians=False)
        if feedback.isCanceled():
            return {}

        aspect_path = self.build_aspect_layer(feedback, context, dem_layer, in_radians=False)
        if feedback.isCanceled():
            return {}

        output_file_path = self.parameterAsFileOutput(parameters, 'IMG_PLOT_OUT_PATH', context)

        computation.statistics.build_polar_diagrams(
            input_layer.source(),
            slope_path,
            aspect_path,
            output_file_path=output_file_path,
            show_plot=False
        )

        return {}
