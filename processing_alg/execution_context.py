import logging
import os
from dataclasses import dataclass
from math import radians, cos
from typing import Dict, Any

import processing
from qgis.core import QgsProcessingContext, QgsProcessingFeedback, QgsRasterLayer

from ..computation.gdal_utils import read_band_as_array
from ..computation.qgis_utils import check_compatible, set_multiprocessing_metadata, qgis_path


@dataclass
class ExecutionContext:
    input_layer_path: str = None
    input_layer_band_count: int = None
    output_file_path: str = None
    sza_degrees: float = None
    solar_azimuth_degrees: float = None
    run_parallel: bool = False
    task_timeout: int = 10000
    keep_in_memory: bool = True
    qgis_dir: str = None
    need_load: bool = False
    _slope_path: str = None
    _aspect_path: str = None
    _luminance_path: str = None

    @property
    def need_qgis_init(self):
        return self.qgis_dir is not None

    @property
    def slope_path(self):
        return self._slope_path

    @property
    def aspect_path(self):
        return self._aspect_path

    @property
    def luminance_path(self):
        return self._luminance_path

    @property
    def luminance_bytes(self):
        if not self.keep_in_memory:
            return read_band_as_array(self._luminance_path, band_idx=1)
        if getattr(self, '_luminance_bytes', None) is None:
            self._luminance_bytes = read_band_as_array(self._luminance_path, band_idx=1)
        return self._luminance_bytes

    def sza_cosine(self):
        return cos(radians(self.sza_degrees))

    def azimuth_cosine(self):
        return cos(radians(self.solar_azimuth_degrees))

    def is_canceled(self):
        return False

    def log(self, message: str):
        pass


# todo use simple raster calc in methods instead of qgis raster calc
class QgisExecutionContext(ExecutionContext):
    def __init__(
            self,
            qgis_context: QgsProcessingContext,
            qgis_feedback: QgsProcessingFeedback,
            qgis_params: Dict[str, Any],
            input_layer: QgsRasterLayer,
            dem_layer: QgsRasterLayer,
            output_file_path: str = None,
            sza_degrees: float = None,
            solar_azimuth_degrees: float = None,
            run_parallel: bool = False,
            task_timeout: int = 10000,
            keep_in_memory: bool = True):
        super().__init__(
            input_layer_path=input_layer.source(),
            input_layer_band_count=input_layer.bandCount(),
            output_file_path=output_file_path,
            sza_degrees=sza_degrees,
            solar_azimuth_degrees=solar_azimuth_degrees,
            run_parallel=run_parallel,
            task_timeout=task_timeout,
            keep_in_memory=keep_in_memory,
            need_load=True
        )
        check_compatible(input_layer, dem_layer)
        self.dem_layer = dem_layer
        self.qgis_context = qgis_context
        self.qgis_feedback = qgis_feedback
        self.qgis_params = qgis_params

    def is_canceled(self):
        return self.qgis_feedback.isCanceled()

    def log(self, message: str):
        return self.qgis_feedback.pushInfo(message)

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
        _ = self.luminance_path
        return super().luminance_bytes

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
class SerializableQgisExecutionContext(ExecutionContext):
    @staticmethod
    def from_ctx(ctx: QgisExecutionContext):
        luminance_path = ctx.luminance_path

        set_multiprocessing_metadata()
        return SerializableQgisExecutionContext(
            input_layer_path=ctx.input_layer_path,
            input_layer_band_count=ctx.input_layer_band_count,
            _slope_path=ctx.slope_path,
            _aspect_path=ctx.aspect_path,
            _luminance_path=luminance_path,
            output_file_path=ctx.output_file_path,
            sza_degrees=ctx.sza_degrees,
            solar_azimuth_degrees=ctx.solar_azimuth_degrees,
            task_timeout=ctx.task_timeout,
            keep_in_memory=ctx.keep_in_memory,
            qgis_dir=qgis_path(),
            need_load=False
        )

    @property
    def qgis_context(self):
        return None

    @property
    def qgis_feedback(self):

        class InnerFeedback(QgsProcessingFeedback):
            def __init__(inner, logFeedback: bool):
                super().__init__(logFeedback)

            def pushInfo(inner, info: str) -> None:
                super().pushInfo(info)
                self.log(info)

        return None

    @property
    def qgis_params(self):
        return dict()

    def log(self, message: str):
        logging.basicConfig(level=logging.INFO,
                            filename=fr"D:\PyCharmProjects\QgisPlugin\log\log-{os.path.basename(self.output_file_path)}.log",
                            filemode="w")
        logging.info(message)
