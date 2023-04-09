import logging
import os
from dataclasses import dataclass, field
from math import radians, cos
from typing import Dict, Any

import processing
from qgis.core import QgsProcessingContext, QgsProcessingFeedback, QgsRasterLayer

from ..computation.gdal_utils import read_band_as_array
from ..computation.qgis_utils import check_compatible
from ..dependencies import qgis_path, set_multiprocessing_metadata


# todo use simple raster calc in methods instead of qgis raster calc
@dataclass
class QgisExecutionContext:
    qgis_context: QgsProcessingContext
    qgis_feedback: QgsProcessingFeedback
    qgis_params: Dict[str, Any]
    # todo replace with input_path and band_count
    input_layer: QgsRasterLayer
    _dem_layer: QgsRasterLayer
    output_file_path: str = None
    sza_degrees: float = None
    solar_azimuth_degrees: float = None
    run_parallel: bool = False
    task_timeout: int = 10000
    keep_in_memory: bool = True
    qgis_path: str = None

    def __post_init__(self):
        check_compatible(self.input_layer, self._dem_layer)

    def is_canceled(self):
        return self.qgis_feedback.isCanceled()

    def log(self, message: str):
        return self.qgis_feedback.pushInfo(message)

    @property
    def input_layer_path(self):
        return self.input_layer.source()

    @property
    def input_layer_band_count(self):
        return self.input_layer.bandCount()

    @property
    def slope_path(self):
        if getattr(self, '_slope_path', None) is None:
            self._slope_path = self.calculate_slope()
        return self._slope_path

    @property
    def aspect_path(self):
        if getattr(self, '_aspect_path', None) is None:
            self._aspect_path = self.calculate_aspect()
        return self._aspect_path

    @property
    def luminance_path(self):
        if getattr(self, '_luminance_path', None) is None:
            self._luminance_path = self.calculate_luminance(
                self.slope_path,
                self.aspect_path
            )
        return self._luminance_path

    @property
    def luminance_bytes(self):
        if not self.keep_in_memory:
            return read_band_as_array(self.luminance_path, band_idx=1)
        if getattr(self, '_luminance_bytes', None) is None:
            self._luminance_bytes = read_band_as_array(self.luminance_path, band_idx=1)
        return self._luminance_bytes

    def sza_cosine(self):
        return cos(radians(self.sza_degrees))

    def azimuth_cosine(self):
        return cos(radians(self.solar_azimuth_degrees))

    def calculate_slope(self, in_radians=True) -> str:
        results = processing.run(
            "gdal:slope",
            {
                'INPUT': self._dem_layer,
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

    def calculate_aspect(self, in_radians=True) -> str:
        results = processing.run(
            "gdal:aspect",
            {
                'INPUT': self._dem_layer,
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


@dataclass
class SerializableCorrectionExecutionContext:
    input_layer_path: str = None
    input_layer_band_count: int = None
    slope_path: str = None,
    aspect_path: str = None,
    luminance_path: str = None,
    output_file_path: str = None
    sza_degrees: float = None
    solar_azimuth_degrees: float = None
    run_parallel: bool = False
    task_timeout: int = 10000
    keep_in_memory: bool = True
    qgis_params: Dict[str, Any] = field(default_factory=dict)
    qgis_path: str = None

    @property
    def qgis_context(self):
        return None

    @property
    def qgis_feedback(self):
        return None

    @staticmethod
    def from_ctx(ctx: QgisExecutionContext):
        luminance_path = ctx.luminance_path

        set_multiprocessing_metadata()
        return SerializableCorrectionExecutionContext(
            input_layer_path=ctx.input_layer_path,
            input_layer_band_count=ctx.input_layer_band_count,
            slope_path=ctx.slope_path,
            aspect_path=ctx.aspect_path,
            luminance_path=luminance_path,
            output_file_path=ctx.output_file_path,
            sza_degrees=ctx.sza_degrees,
            solar_azimuth_degrees=ctx.solar_azimuth_degrees,
            run_parallel=ctx.run_parallel,
            task_timeout=ctx.task_timeout,
            keep_in_memory=ctx.keep_in_memory,
            qgis_path=qgis_path()
        )

    @property
    def luminance_bytes(self):
        if not self.keep_in_memory:
            return read_band_as_array(self.luminance_path, band_idx=1)
        if getattr(self, '_luminance_bytes', None) is None:
            self._luminance_bytes = read_band_as_array(self.luminance_path, band_idx=1)
        return self._luminance_bytes

    def sza_cosine(self):
        return cos(radians(self.sza_degrees))

    def is_canceled(self):
        return False

    def log(self, message: str):
        logging.basicConfig(level=logging.INFO,
                            filename=fr"D:\PyCharmProjects\QgisPlugin\log\log-{os.path.basename(self.output_file_path)}.log",
                            filemode="w")
        logging.info(message)
