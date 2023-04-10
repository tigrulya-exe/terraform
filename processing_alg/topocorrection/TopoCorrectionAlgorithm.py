import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any

import processing
from qgis.core import QgsProcessingException

from ..execution_context import QgisExecutionContext, SerializableCorrectionExecutionContext
from ...computation.raster_calc import SimpleRasterCalc, RasterInfo


class TopoCorrectionAlgorithm:
    def __init__(self):
        self.calc = SimpleRasterCalc()
        self.salt = random.randint(1, 100000)

    @staticmethod
    def get_name():
        pass

    def init(self, ctx: QgisExecutionContext):
        pass

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        pass

    def process(self, ctx: QgisExecutionContext) -> Dict[str, Any]:

        self.init(ctx)

        ctx.log(f"{self.get_name()} correction started: parallel={ctx.run_parallel}")
        result_bands = self._process_parallel(ctx) if ctx.run_parallel else self._process_sequentially(ctx)

        ctx.log(f"start merge results for {self.get_name()}")

        results = processing.run(
            "gdal:merge",
            {
                'INPUT': result_bands,
                'PCT': False,
                'SEPARATE': True,
                'DATA_TYPE': 5,
                'OUTPUT': ctx.output_file_path
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        ctx.log(f"end merge results for {self.get_name()}")

        return results

    def _process_parallel(self, ctx: QgisExecutionContext):
        result_bands = []

        serializable_ctx = SerializableCorrectionExecutionContext.from_ctx(ctx)

        futures = []

        # todo handle exceptions from executor
        with ProcessPoolExecutor() as executor:
            for band_idx in range(ctx.input_layer_band_count):
                future = executor.submit(_process_band_wrapper, self, serializable_ctx, band_idx)
                futures.append(future)

                if ctx.is_canceled():
                    [future.cancel() for future in futures]
                    executor.shutdown(cancel_futures=True)
                    return None

                result_bands.append(self.output_file_path(str(band_idx)))

            for band_idx, future in enumerate(futures):
                future.result(timeout=ctx.task_timeout)
                ctx.log(f"Task for band {band_idx + 1} finished")

        return result_bands

    def _process_sequentially(self, ctx: QgisExecutionContext):
        result_bands = []

        for band_idx in range(ctx.input_layer_band_count):
            try:
                result = self.process_band(ctx, band_idx)
                result_bands.append(result)
            except QgsProcessingException as exc:
                raise RuntimeError(f"Error during performing topocorrection: {exc}")

            if ctx.is_canceled():
                return None

        return result_bands

    def output_file_path(self, postfix=''):
        return os.path.join(
            tempfile.gettempdir(),
            f'{self.get_name()}_{self.salt}_{postfix}.tif'
        )

    def raster_calculate(self, ctx: QgisExecutionContext, calc_func, raster_infos: list[RasterInfo],
                         out_file_postfix='',
                         **kwargs):
        if ctx.is_canceled():
            raise RuntimeError("Canceled")

        out_path = self.output_file_path(out_file_postfix)

        calc_start = time.process_time_ns()
        self.calc.calculate(
            func=calc_func,
            output_path=out_path,
            raster_infos=raster_infos,
            **kwargs
        )
        calc_end = time.process_time_ns()

        ctx.log(f"Time for band {raster_infos[0].band}: {(calc_end - calc_start) / 1000000} ms")

        return out_path


def _process_band_wrapper(algorithm, _ctx, _band_idx):
    algorithm.process_band(_ctx, _band_idx)
