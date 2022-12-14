import os
import random
import tempfile
import textwrap
from math import cos, radians
from typing import Dict, Any

import numpy as np
import processing
from qgis._core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback
from qgis.core import (QgsProcessingException)

from computation.my_simple_calc import SimpleRasterCalc, RasterInfo


class TopoCorrectionContext:
    def __init__(
            self,
            qgis_context: QgsProcessingContext,
            qgis_feedback: QgsProcessingFeedback,
            qgis_params: Dict[str, Any],
            input_layer: QgsRasterLayer,
            slope_rad_path: str,
            aspect_path: str,
            luminance_path: str,
            solar_zenith_angle: float,
            solar_azimuth: float):
        self.qgis_context = qgis_context
        self.qgis_params = qgis_params
        self.qgis_feedback = qgis_feedback
        self.input_layer = input_layer
        self.slope_rad_path = slope_rad_path
        self.aspect_path = aspect_path
        self.luminance_path = luminance_path
        self.solar_zenith_angle = solar_zenith_angle
        self.solar_azimuth = solar_azimuth

    def sza_cosine(self):
        return cos(radians(self.solar_zenith_angle))

    def azimuth_cosine(self):
        return cos(radians(self.solar_azimuth))


class TopoCorrectionAlgorithm:
    def __init__(self):
        self.calc = SimpleRasterCalc()

    @staticmethod
    def get_name():
        pass

    def init(self, ctx: TopoCorrectionContext):
        pass

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        pass

    def safe_divide(self, top: str, bottom: str) -> str:
        return self.safe_divide_check(top, bottom, bottom)

    def safe_divide_check(self, top, bottom, null_check):
        return np.divide(
            top,
            bottom,
            out=np.zeros_like(bottom, dtype='float32'),
            where=null_check != 0
        )

    def output_file_path(self, postfix=''):
        return os.path.join(
            tempfile.gettempdir(),
            f'{self.get_name()}_{random.randint(1, 100000)}_{postfix}'
        )

    def calculate(self, calc_func, raster_infos: list[RasterInfo]):
        out_path = self.output_file_path()
        self.calc.calculate(
            func=calc_func,
            output_path=out_path,
            raster_infos=raster_infos
        )
        return out_path

    def process(self, ctx: TopoCorrectionContext) -> Dict[str, Any]:
        self.init(ctx)
        result_bands = []

        for band_idx in range(ctx.input_layer.bandCount()):
            try:
                result = self.process_band(ctx, band_idx)
                result_bands.append(result)
            except QgsProcessingException as exc:
                raise RuntimeError(f"Error during performing topocorrection: {exc}")

            if ctx.qgis_feedback.isCanceled():
                # todo
                return {}

        return processing.runAndLoadResults(
            "gdal:merge",
            {
                'INPUT': result_bands,
                'PCT': False,
                'SEPARATE': True,
                'NODATA_INPUT': None,
                'NODATA_OUTPUT': None,
                'OPTIONS': '',
                'EXTRA': '',
                'DATA_TYPE': 5,
                'OUTPUT': ctx.qgis_params['OUTPUT']
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
