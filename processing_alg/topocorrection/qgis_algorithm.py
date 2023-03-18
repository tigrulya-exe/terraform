# -*- coding: utf-8 -*-

"""
/***************************************************************************
 TerraformTopoCorrection
                                 A QGIS plugin
 Topographically correct provided input layer.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2023-02-27
        copyright            : (C) 2023 by Tigran Manasyan
        email                : t.manasyan@g.nsu.ru
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Tigran Manasyan'
__date__ = '2023-02-27'
__copyright__ = '(C) 2023 by Tigran Manasyan'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from enum import Enum
from typing import Dict, Any

from qgis.PyQt.QtCore import QCoreApplication
from qgis._core import QgsProcessingParameterBoolean
from qgis.core import (QgsProcessingContext,
                       QgsProcessingFeedback,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination)

from .CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm
from .CosineCTopoCorrectionAlgorithm import CosineCTopoCorrectionAlgorithm
from .CosineTTopoCorrectionAlgorithm import CosineTTopoCorrectionAlgorithm
from .MinnaertScsTopoCorrectionAlgorithm import MinnaertScsTopoCorrectionAlgorithm
from .MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from .PbcTopoCorrectionAlgorithm import PbcTopoCorrectionAlgorithm
from .PbmTopoCorrectionAlgorithm import PbmTopoCorrectionAlgorithm
from .ScsCTopoCorrectionAlgorithm import ScsCTopoCorrectionAlgorithm
from .ScsTopoCorrectionAlgorithm import ScsTopoCorrectionAlgorithm
from .TeilletRegressionTopoCorrectionAlgorithm import TeilletRegressionTopoCorrectionAlgorithm
from .TopoCorrectionAlgorithm import TopoCorrectionContext
from .VecaTopoCorrectionAlgorithm import VecaTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ..terraform_algorithm import TerraformProcessingAlgorithm
from ...computation.qgis_utils import add_layer_to_load


# from processing.core.ProcessingConfig import ProcessingConfig
# from processing.script import ScriptUtils
# print(ProcessingConfig.getSetting('SCRIPTS_FOLDERS'))
# print(ScriptUtils.defaultScriptsFolder())
class TerraformTopoCorrectionAlgorithm(TerraformProcessingAlgorithm):
    class AuxiliaryLayers(Enum):
        ASPECT = 0
        SLOPE = 1
        LUMINANCE = 2

    def __init__(self):
        super().__init__()
        # todo dynamically scan directory
        self.algorithms = self.find_algorithms()

    def find_algorithms(self):
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
        return TerraformTopoCorrectionAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'terraform_topocorrection'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Topographically correct raster image')

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Topographically corrects provided raster image. '
                       'Algorithm automatically builds slope and aspect layers for input image for calculation of '
                       'the incidence angle between the sun and normal to the ground surface. '
                       'The latter then is used in chosen topographic correction algorithm.\n'
                       r'<b>Note:</b> currently, the input raster image and the DEM must have the same CRS, '
                       'extent and spatial resolution.\n'
                       """Implemented algorithms: 
                       <a href="https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028">Cosine-T</a>
                       <a href="https://www.asprs.org/wp-content/uploads/pers/1989journal/sep/1989_sep_1303-1309.pdf">Cosine-C</a>
                       <a href="https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028">C-correction</a>
                       <a href="http://dx.doi.org/10.1016/S0034-4257(97)00177-6">SCS</a>
                       <a href="http://dx.doi.org/10.1109/TGRS.2005.852480">SCS+C</a>
                       <a href="https://www.asprs.org/wp-content/uploads/pers/1980journal/sep/1980_sep_1183-1189.pdf">Minnaert</a>
                       <a href="https://ui.adsabs.harvard.edu/abs/2002PhDT........92R/abstract">Minnaert-SCS</a>
                       <a href="https://www.researchgate.net/publication/235244169_Pixel-based_Minnaert_Correction_Method_for_Reducing_Topographic_Effects_on_a_Landsat_7_ETM_Image">Pixel-based Minnaert</a> 
                       <a href="https://www.tandfonline.com/doi/full/10.1080/01431160701881889">Pixel-based C correction</a>
                       <a href="https://ieeexplore.ieee.org/abstract/document/4423917/">VECA</a>
                       <a href="https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028">Teillet regression</a>
                       """)

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
                self.tr('Topographic correction algorithm'),
                options=self.algorithms.keys(),
                allowMultiple=False,
                defaultValue=CTopoCorrectionAlgorithm.get_name(),
                usesStaticStrings=True
            )
        )

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
            QgsProcessingParameterRasterDestination(
                'OUTPUT',
                self.tr('Result output')
            )
        )

        show_layers_param = QgsProcessingParameterEnum(
            'SHOW_AUXILIARY_LAYERS',
            self.tr('Auxiliary generated layers to open after running algorithm'),
            options=[e.name for e in self.AuxiliaryLayers],
            allowMultiple=True,
            optional=True
        )
        self._additional_param(show_layers_param)

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

        # todo migrate topo correction algorithms to QgisExecutionContext
        class TopoCorrectionQgisExecutionContext(QgisExecutionContext):
            def __init__(inner):
                super().__init__(context, feedback, parameters, input_layer, dem_layer, sza_degrees=solar_zenith_angle,
                                 solar_azimuth_degrees=solar_azimuth)

            def calculate_slope(inner, in_radians=True) -> str:
                result_path = super().calculate_slope(in_radians)
                self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.SLOPE, "Slope_gen")
                return result_path

            def calculate_aspect(inner, in_radians=True) -> str:
                result_path = super().calculate_aspect(in_radians)
                self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.ASPECT, "Aspect_gen")
                return result_path

            def calculate_luminance(inner, slope_path=None, aspect_path=None) -> str:
                result_path = super().calculate_luminance(slope_path, aspect_path)
                self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.LUMINANCE, "Luminance_gen")
                return result_path

        # todo tmp solution, need migrate fully to execution_context
        exec_ctx = TopoCorrectionQgisExecutionContext()

        self.show_tmp_layers = self.parameterAsEnums(parameters, 'SHOW_AUXILIARY_LAYERS', context)
        run_parallel = self.parameterAsBoolean(parameters, 'RUN_PARALLEL', context)
        task_timeout = self.parameterAsInt(parameters, 'TASK_TIMEOUT', context)

        slope_rad_path = exec_ctx.calculate_slope(in_radians=True)
        if feedback.isCanceled():
            return {}

        aspect_path = exec_ctx.calculate_aspect(in_radians=True)
        if feedback.isCanceled():
            return {}

        luminance_path = exec_ctx.calculate_luminance(slope_rad_path, aspect_path)
        if feedback.isCanceled():
            return {}

        # todo replace with exec_ctx
        topo_context = TopoCorrectionContext(
            qgis_context=context,
            qgis_feedback=feedback,
            qgis_params=parameters,
            input_layer=input_layer,
            slope_rad_path=slope_rad_path,
            aspect_path=aspect_path,
            luminance_path=luminance_path,
            solar_zenith_angle=solar_zenith_angle,
            solar_azimuth=solar_azimuth,
            run_parallel=run_parallel,
            task_timeout=task_timeout
        )

        tc_algorithm_name = self.parameterAsEnumString(parameters, 'TOPO_CORRECTION_ALGORITHM', context)

        # add validation
        return self.algorithms[tc_algorithm_name].process(topo_context)

    def _add_layer_to_project(self, context, layer_path, show_label: AuxiliaryLayers, name="out"):
        if show_label.value in self.show_tmp_layers:
            add_layer_to_load(context, layer_path, name)
