import functools
from math import radians
from typing import Dict, Any

import processing
from qgis.core import QgsProcessingContext, QgsProcessingFeedback, QgsRasterLayer

class QgisExecutionContext:
    def __init__(
            self,
            qgis_context: QgsProcessingContext,
            qgis_feedback: QgsProcessingFeedback,
            qgis_params: Dict[str, Any],
            input_layer: QgsRasterLayer,
            dem_layer: QgsRasterLayer,
            output_file_path: str = None,
            sza_degrees: float = None,
            solar_azimuth_degrees: float = None):
        self.qgis_context = qgis_context
        self.qgis_params = qgis_params
        self.qgis_feedback = qgis_feedback
        self.input_layer = input_layer
        self.dem_layer = dem_layer
        self.sza_degrees = sza_degrees
        self.solar_azimuth_degrees = solar_azimuth_degrees
        self.output_file_path = output_file_path

    @functools.cache
    def calculate_slope(self, in_radians=True) -> str:
        results = processing.run(
            "gdal:slope",
            {
                'INPUT': self.dem_layer,
                'BAND': 1,
                # magic number 111120 lol
                'SCALE': 1,
                'AS_PERCENT': False,
                'COMPUTE_EDGES': True,
                'ZEVENBERGEN': True,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            },
            feedback=self.qgis_feedback,
            context=self.qgis_context,
            is_child_algorithm=True
        )
        result_deg_path = results['OUTPUT']

        if not in_radians:
            return result_deg_path

        slope_cos_result = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': result_deg_path,
                'BAND_A': 1,
                'FORMULA': f'deg2rad(A)',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=self.qgis_feedback,
            context=self.qgis_context
        )
        return slope_cos_result['OUTPUT']

    @functools.cache
    def calculate_aspect(self, in_radians=True) -> str:
        results = processing.run(
            "gdal:aspect",
            {
                'INPUT': self.dem_layer,
                'BAND': 1,
                'TRIG_ANGLE': False,
                'ZERO_FLAT': True,
                'COMPUTE_EDGES': True,
                'ZEVENBERGEN': True,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            },
            feedback=self.qgis_feedback,
            context=self.qgis_context,
            is_child_algorithm=True
        )

        result_deg_path = results['OUTPUT']
        if not in_radians:
            return result_deg_path

        slope_cos_result = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': result_deg_path,
                'BAND_A': 1,
                'FORMULA': f'deg2rad(A)',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=self.qgis_feedback,
            context=self.qgis_context
        )
        return slope_cos_result['OUTPUT']

    @functools.cache
    def calculate_luminance(self, slope_path=None, aspect_path=None) -> str:
        if self.sza_degrees is None or self.solar_azimuth_degrees is None:
            raise RuntimeError(f"SZA and azimuth angles not initializes")

        if slope_path is None:
            slope_path = self.calculate_slope()

        if aspect_path is None:
            aspect_path = self.calculate_aspect()

        sza_radians = radians(self.sza_degrees)
        solar_azimuth_radians = radians(self.solar_azimuth_degrees)

        results = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': slope_path,
                'BAND_A': 1,
                'INPUT_B': aspect_path,
                'BAND_B': 1,
                'FORMULA': f'fmax(0.0, (cos({sza_radians})*cos(A) + '
                           f'sin({sza_radians})*sin(A)*cos(B - {solar_azimuth_radians})))',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=self.qgis_feedback,
            context=self.qgis_context,
            is_child_algorithm=True
        )
        result_path = results['OUTPUT']
        return result_path
