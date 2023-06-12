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

import random
from enum import Enum
from typing import Dict, Any

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingContext,
                       QgsProcessingFeedback,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination)

from .CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm
from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from .TopoCorrectionPostProcessor import TopoCorrectionPostProcessor
from ..ParallelProcessingParamMixin import ParallelProcessingParamMixin
from ..execution_context import QgisExecutionContext
from ..terraform_algorithm import TerraformProcessingAlgorithm
from ..topocorrection import DEFAULT_CORRECTIONS
from ...util.qgis_utils import add_layer_to_load, get_project_tmp_dir


class TerraformTopoCorrectionAlgorithm(TerraformProcessingAlgorithm, ParallelProcessingParamMixin):
    class AuxiliaryLayers(Enum):
        ASPECT = 0
        SLOPE = 1
        LUMINANCE = 2

    def __init__(self):
        super().__init__()
        # todo dynamically scan directory
        self.algorithms = self._find_algorithms()

    def _find_algorithms(self) -> dict[str, TopoCorrectionAlgorithm]:
        algorithms = DEFAULT_CORRECTIONS
        algorithms_dict = dict()
        for algorithm in algorithms:
            algorithms_dict[algorithm.name()] = algorithm()
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
                       "Implemented algorithms:\n"
                       + '\n'.join([algorithm.description() for algorithm in self.algorithms.values()]))

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
                defaultValue=CTopoCorrectionAlgorithm.name(),
                usesStaticStrings=True
            )
        )

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

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                'OUTPUT',
                self.tr('Result output'),
                createByDefault=True
            )
        )

        ignore_threshold_param = QgsProcessingParameterNumber(
            'PIXEL_IGNORE_THRESHOLD',
            self.tr('Upper limit of the pixel value to ignore. All points with a lower value will not be corrected'),
            defaultValue=5.0,
            type=QgsProcessingParameterNumber.Double
        )
        self._additional_param(ignore_threshold_param)

        show_layers_param = QgsProcessingParameterEnum(
            'SHOW_AUXILIARY_LAYERS',
            self.tr('Auxiliary generated layers to open after running algorithm'),
            options=[e.name for e in self.AuxiliaryLayers],
            allowMultiple=True,
            optional=True
        )
        self._additional_param(show_layers_param)

        params = self.parallel_run_params()
        [self._additional_param(param) for param in params]

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
        run_parallel = self.get_run_parallel_param(parameters, context)
        task_timeout = self.get_parallel_timeout_param(parameters, context)
        worker_count = self.get_worker_count_param(parameters, context)

        output_file_path = self.parameterAsFileOutput(parameters, 'OUTPUT', context)
        pixel_ignore_threshold = self.parameterAsDouble(parameters, 'PIXEL_IGNORE_THRESHOLD', context)

        class TopoCorrectionQgisExecutionContext(QgisExecutionContext):
            def __init__(inner):
                super().__init__(
                    qgis_context=context,
                    qgis_feedback=feedback,
                    qgis_params=parameters,
                    input_layer=input_layer,
                    dem_layer=dem_layer,
                    output_file_path=output_file_path,
                    sza_degrees=solar_zenith_angle,
                    solar_azimuth_degrees=solar_azimuth,
                    run_parallel=run_parallel,
                    task_timeout=task_timeout,
                    worker_count=worker_count,
                    pixel_ignore_threshold=pixel_ignore_threshold,
                    tmp_dir=get_project_tmp_dir()
                )
                inner.salt = random.randint(1, 100000)

            def calculate_slope(inner, in_radians=True) -> str:
                result_path = super().calculate_slope(in_radians)
                self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.SLOPE, f"Slope_{inner.salt}")
                return result_path

            def calculate_aspect(inner, in_radians=True) -> str:
                result_path = super().calculate_aspect(in_radians)
                self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.ASPECT, f"Aspect_{inner.salt}")
                return result_path

            def calculate_luminance(inner, slope_path=None, aspect_path=None) -> str:
                result_path = super().calculate_luminance(slope_path, aspect_path)
                self._add_layer_to_project(context, result_path, self.AuxiliaryLayers.LUMINANCE,
                                           f"Luminance_{inner.salt}")
                return result_path

        ctx = TopoCorrectionQgisExecutionContext()
        correction_name = self.parameterAsEnumString(parameters, 'TOPO_CORRECTION_ALGORITHM', context)

        corrected_img_path = self.algorithms[correction_name].process(ctx)

        # todo: change band names even without load on completion
        if context.willLoadLayerOnCompletion(corrected_img_path):
            context.layerToLoadOnCompletionDetails(
                corrected_img_path).setPostProcessor(TopoCorrectionPostProcessor.create(input_layer))

        return {"OUTPUT": corrected_img_path}

    def _add_layer_to_project(self, context, layer_path, show_label: AuxiliaryLayers, name="out"):
        if show_label.value in self.show_tmp_layers:
            add_layer_to_load(context, layer_path, name)
