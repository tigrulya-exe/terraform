from enum import Enum
from math import radians
from typing import Dict, Any

import processing
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingAlgorithm,
                       QgsProcessingContext,
                       QgsProcessingFeedback,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination)

from computation.qgis_utils import add_layer_to_project
from topocorrection.CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm
from topocorrection.TopoCorrectionAlgorithm import TopoCorrectionContext

class RoseDiagramsEvaluationAlgorithm(QgsProcessingAlgorithm):
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
        return 'terraform_plottest'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Plot test')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr('NSU')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs
        to.
        """
        return 'nsu'

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Topographically correct provided input layer.')

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
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Solar zenith angle'),
                defaultValue=57.2478878065826,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar azimuth'),
                defaultValue=177.744663052425,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                'OUTPUT',
                self.tr('Raster output')
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
        solar_zenith_angle = self.parameterAsDouble(parameters, 'SZA', context)
        solar_azimuth = self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context)

        self.show_tmp_layers = self.parameterAsEnums(parameters, 'SHOW_AUXILIARY_LAYERS', context)
        feedback.pushInfo(f"ssssss {self.show_tmp_layers}")

        slope_rad_path = self.build_slope_rad_layer(feedback, context, dem_layer)
        if feedback.isCanceled():
            return {}

        aspect_path = self.build_aspect_layer(feedback, context, dem_layer)
        if feedback.isCanceled():
            return {}

        luminance_path = self.compute_luminance(
            feedback, context, slope_rad_path, aspect_path, solar_zenith_angle, solar_azimuth)

        if feedback.isCanceled():
            return {}

        tc_algorithm_name = self.parameterAsEnumString(parameters, 'TOPO_CORRECTION_ALGORITHM', context)

        # add validation
        return self.algorithms[tc_algorithm_name].process(topo_context)
