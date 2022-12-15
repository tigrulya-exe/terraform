from enum import Enum, auto
from math import radians
from typing import Dict, Any

import processing
from qgis.PyQt.QtCore import QCoreApplication
from qgis._core import QgsProcessingContext, QgsProcessingFeedback, \
    QgsProcessingParameterEnum
from qgis.core import (QgsProcessingAlgorithm,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination)

from algorithms.TopoCorrectionAlgorithm import TopoCorrectionContext
from computation.qgis_utils import add_layer_to_project
from topocorrection.CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm


# from processing.core.ProcessingConfig import ProcessingConfig
# from processing.script import ScriptUtils
# print(ProcessingConfig.getSetting('SCRIPTS_FOLDERS'))
# print(ScriptUtils.defaultScriptsFolder())
class ExampleProcessingAlgorithm(QgsProcessingAlgorithm):
    class AuxiliaryLayers(Enum):
        ASPECT = 0
        SLOPE = 1
        LUMINANCE = 2

    def __init__(self):
        super().__init__()
        # todo dynamically scan directory
        self.algorithms = self.find_algorithms()

    def find_algorithms(self):
        from topocorrection.CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm
        from topocorrection.CosineCTopoCorrectionAlgorithm import CosineCTopoCorrectionAlgorithm
        from topocorrection.CosineTTopoCorrectionAlgorithm import CosineTTopoCorrectionAlgorithm
        from topocorrection.MinnaertScsTopoCorrectionAlgorithm import MinnaertScsTopoCorrectionAlgorithm
        from topocorrection.MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
        from topocorrection.PbmTopoCorrectionAlgorithm import PbmTopoCorrectionAlgorithm
        from topocorrection.ScsCTopoCorrectionAlgorithm import ScsCTopoCorrectionAlgorithm
        from topocorrection.ScsTopoCorrectionAlgorithm import ScsTopoCorrectionAlgorithm
        from topocorrection.TeilletRegressionTopoCorrectionAlgorithm import TeilletRegressionTopoCorrectionAlgorithm
        from topocorrection.VecaTopoCorrectionAlgorithm import VecaTopoCorrectionAlgorithm
        from topocorrection.PbcTopoCorrectionAlgorithm import PbcTopoCorrectionAlgorithm
        algorithms = [
            CosineTTopoCorrectionAlgorithm,
            CosineCTopoCorrectionAlgorithm,
            CTopoCorrectionAlgorithm,
            ScsTopoCorrectionAlgorithm,
            ScsCTopoCorrectionAlgorithm,
            MinnaertTopoCorrectionAlgorithm,
            MinnaertScsTopoCorrectionAlgorithm,
            PbmTopoCorrectionAlgorithm,
            VecaTopoCorrectionAlgorithm,
            TeilletRegressionTopoCorrectionAlgorithm,
            PbcTopoCorrectionAlgorithm
        ]
        algorithms_dict = dict()
        for algorithm in algorithms:
            algorithms_dict[algorithm.get_name()] = algorithm()
        return algorithms_dict

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        # Must return a new copy of your algorithm.
        return ExampleProcessingAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'topographic_correction'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Topographic Correction')

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
            QgsProcessingParameterEnum(
                'TOPO_CORRECTION_ALGORITHM',
                self.tr('Topological correction algorithm'),
                options=self.algorithms.keys(),
                allowMultiple=False,
                defaultValue=CTopoCorrectionAlgorithm.get_name(),
                usesStaticStrings=True
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                'SHOW_AUXILIARY_LAYERS',
                self.tr('Auxiliary layers to show'),
                options=[e.name for e in self.AuxiliaryLayers],
                allowMultiple=True,
                optional=True
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

        topo_context = TopoCorrectionContext(
            qgis_context=context,
            qgis_feedback=feedback,
            qgis_params=parameters,
            input_layer=input_layer,
            slope_rad_path=slope_rad_path,
            aspect_path=aspect_path,
            luminance_path=luminance_path,
            solar_zenith_angle=solar_zenith_angle,
            solar_azimuth=solar_azimuth
        )

        tc_algorithm_name = self.parameterAsEnumString(parameters, 'TOPO_CORRECTION_ALGORITHM', context)

        # add validation
        return self.algorithms[tc_algorithm_name].process(topo_context)

    def build_slope_rad_layer(self, feedback, context, dem_layer) -> (str, str):
        slope_path = self.build_slope_layer(feedback, context, dem_layer)

        slope_cos_result = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': slope_path,
                'BAND_A': 1,
                'FORMULA': f'deg2rad(A)',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=feedback,
            context=context
        )
        return slope_cos_result['OUTPUT']

    def build_slope_layer(self, feedback, context, dem_layer) -> str:
        results = processing.run(
            "gdal:slope",
            {
                'INPUT': dem_layer,
                'BAND': 1,
                # magic number 111120 lol
                'SCALE': 1,
                'AS_PERCENT': False,
                'COMPUTE_EDGES': True,
                'ZEVENBERGEN': True,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            },
            feedback=feedback,
            context=context,
            is_child_algorithm=True
        )

        result_path = results['OUTPUT']
        self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.SLOPE, "Slope_gen")

        return result_path

    def build_aspect_layer(self, feedback, context, dem_layer) -> str:
        results = processing.run(
            "gdal:aspect",
            {
                'INPUT': dem_layer,
                'BAND': 1,
                'TRIG_ANGLE': False,
                'ZERO_FLAT': True,
                'COMPUTE_EDGES': True,
                'ZEVENBERGEN': True,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            },
            feedback=feedback,
            context=context,
            is_child_algorithm=True
        )

        result_path = results['OUTPUT']
        self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.ASPECT, "Aspect_gen")

        return result_path

    def compute_luminance(self, feedback, context, slope_rad_path: str, aspect_path: str, sza: float,
                          solar_azimuth: float) -> str:
        sza_radians = radians(sza)
        solar_azimuth_radians = radians(solar_azimuth)

        results = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': slope_rad_path,
                'BAND_A': 1,
                'INPUT_B': aspect_path,
                'BAND_B': 1,
                'FORMULA': f'fmax(0.0, (cos({sza_radians})*cos(A) + '
                           f'sin({sza_radians})*sin(A)*cos(deg2rad(B) - {solar_azimuth_radians})))',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=feedback,
            context=context,
            is_child_algorithm=True
        )
        result_path = results['OUTPUT']
        self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.LUMINANCE, "Luminance_gen")
        return result_path

    def _add_layer_to_project(self, context, layer_path, show_label: AuxiliaryLayers, name="out"):
        if show_label.value in self.show_tmp_layers:
            add_layer_to_project(context, layer_path, name)
