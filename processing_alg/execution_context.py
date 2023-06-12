import logging
import os
import random
import tempfile
from dataclasses import dataclass
from math import radians, cos, sin
from pathlib import Path
from typing import Dict, Any

import numpy as np
import processing
from qgis.core import QgsProcessingContext, QgsProcessingFeedback, QgsRasterLayer

from ..util.gdal_utils import read_band_as_array, get_raster_type_ordinal
from ..util.qgis_utils import check_compatible, set_multiprocessing_metadata, qgis_path, SilentFeedbackWrapper
from ..util.raster_calc import SimpleRasterCalc, RasterInfo


@dataclass
class ExecutionContext:
    input_layer_path: str = None
    input_layer_band_count: int = None
    output_file_path: str = None
    sza_degrees: float = None
    solar_azimuth_degrees: float = None
    run_parallel: bool = False
    task_timeout: int = 10000
    worker_count: int = None
    keep_in_memory: bool = True
    qgis_dir: str = None
    tmp_dir: str = tempfile.gettempdir()
    need_load: bool = False
    _slope_path: str = None
    _aspect_path: str = None
    _luminance_path: str = None
    pixel_ignore_threshold: int = 5

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
    def input_file_name(self) -> str:
        return Path(self.input_layer_path).stem

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

    def log_debug(self, message: str):
        pass

    def log_info(self, message: str):
        pass

    def log_warn(self, message: str):
        pass

    def log_error(self, message: str, fatal=False):
        pass

    def merge_bands(self, band_paths: list[str], gdal_out_type: str, out_path: str = None):
        pass


class QgisExecutionContext(ExecutionContext):
    def __init__(
            self,
            qgis_context: QgsProcessingContext,
            qgis_feedback: QgsProcessingFeedback,
            qgis_params: Dict[str, Any],
            input_layer: QgsRasterLayer,
            dem_layer: QgsRasterLayer,
            tmp_dir: str = tempfile.gettempdir(),
            output_file_path: str = None,
            sza_degrees: float = None,
            solar_azimuth_degrees: float = None,
            run_parallel: bool = False,
            task_timeout: int = 10000,
            worker_count: int = None,
            pixel_ignore_threshold: float = 5,
            keep_in_memory: bool = True):
        super().__init__(
            input_layer_path=input_layer.source(),
            input_layer_band_count=input_layer.bandCount(),
            output_file_path=output_file_path,
            sza_degrees=sza_degrees,
            solar_azimuth_degrees=solar_azimuth_degrees,
            run_parallel=run_parallel,
            task_timeout=task_timeout,
            worker_count=worker_count,
            keep_in_memory=keep_in_memory,
            need_load=True,
            pixel_ignore_threshold=pixel_ignore_threshold,
            tmp_dir=tmp_dir
        )
        check_compatible(input_layer, dem_layer)
        self.dem_layer = dem_layer
        self.qgis_context = qgis_context
        self.qgis_feedback = qgis_feedback
        self.silent_feedback = SilentFeedbackWrapper(qgis_feedback)
        self.qgis_params = qgis_params
        self.calc = SimpleRasterCalc()

    def is_canceled(self):
        return self.qgis_feedback.isCanceled()

    def log_debug(self, message: str):
        return self.qgis_feedback.pushDebugInfo(message)

    def log_info(self, message: str):
        return self.qgis_feedback.pushInfo(message)

    def log_warn(self, message: str):
        return self.qgis_feedback.pushWarning(message)

    def log_error(self, message: str, fatal=False):
        return self.qgis_feedback.reportError(message, fatal)

    def force_cancel(self, error: Exception = None):
        raise error or RuntimeError("Canceled")

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
        self.log_info("[QGIS processing]: calculating slope.")
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
            feedback=self.silent_feedback,
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
            feedback=self.silent_feedback,
            context=self.qgis_context
        )
        return slope_cos_result['OUTPUT']

    def calculate_aspect(self, in_radians=True) -> str:
        self.log_info("[QGIS processing]: calculating aspect.")
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
            feedback=self.silent_feedback,
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
            feedback=self.silent_feedback,
            context=self.qgis_context
        )
        return slope_cos_result['OUTPUT']

    def calculate_luminance(self, slope_path=None, aspect_path=None) -> str:
        self.log_info("[QGIS processing]: calculating luminance.")

        if self.sza_degrees is None or self.solar_azimuth_degrees is None:
            raise RuntimeError(f"SZA and azimuth angles not initializes")

        if slope_path is None:
            slope_path = self.calculate_slope()

        if aspect_path is None:
            aspect_path = self.calculate_aspect()

        sza_radians = radians(self.sza_degrees)
        solar_azimuth_radians = radians(self.solar_azimuth_degrees)

        result_path = os.path.join(self.tmp_dir, f"luminance_{random.randint(0, 9999)}.tif")

        def calc_function(slope, aspect):
            return np.fmax(
                0.0,
                cos(sza_radians) * np.cos(slope) +
                sin(sza_radians) * np.sin(slope) * np.cos(aspect - solar_azimuth_radians))

        self.calc.calculate(
            calc_function,
            result_path,
            [RasterInfo("slope", slope_path),
             RasterInfo("aspect", aspect_path)]
        )

        return result_path

    def merge_bands(self, band_paths: list[str], gdal_out_type: str, out_path: str = None):
        self.log_info("[QGIS processing]: merging bands.")

        out_type_ordinal = get_raster_type_ordinal(gdal_out_type)
        processing_func = processing.runAndLoadResults if self.need_load else processing.run
        merge_results = processing_func(
            "gdal:merge",
            {
                'INPUT': band_paths,
                'PCT': False,
                'SEPARATE': True,
                'DATA_TYPE': out_type_ordinal,
                'OUTPUT': out_path or self.output_file_path
            },
            feedback=self.silent_feedback,
            context=self.qgis_context
        )
        return merge_results['OUTPUT']


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
            worker_count=ctx.worker_count,
            keep_in_memory=ctx.keep_in_memory,
            qgis_dir=qgis_path(),
            tmp_dir=ctx.tmp_dir,
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
                self.log_info(info)

        return None

    @property
    def qgis_params(self):
        return dict()

    def log_info(self, message: str):
        # TODO
        # logging.basicConfig(level=logging.INFO,
        #                     filename=fr"D:\PyCharmProjects\QgisPlugin\log\log-{os.path.basename(self.output_file_path)}.log",
        #                     filemode="w")
        logging.info(message)

    def merge_bands(self, band_paths: list[str], gdal_out_type: str, out_path: str = None):
        self.log_info("[QGIS processing]: merging bands.")

        out_type_ordinal = get_raster_type_ordinal(gdal_out_type)
        processing_func = processing.runAndLoadResults if self.need_load else processing.run
        merge_results = processing_func(
            "gdal:merge",
            {
                'INPUT': band_paths,
                'PCT': False,
                'SEPARATE': True,
                'DATA_TYPE': out_type_ordinal,
                'OUTPUT': out_path or self.output_file_path
            },
            feedback=self.qgis_feedback,
            context=self.qgis_context
        )
        return merge_results['OUTPUT']
