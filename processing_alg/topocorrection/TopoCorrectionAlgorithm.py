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

import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

from ..execution_context import QgisExecutionContext, SerializableQgisExecutionContext
from ...util.gdal_utils import get_raster_type
from ...util.raster_calc import SimpleRasterCalc, RasterInfo


class TopoCorrectionAlgorithm:
    def __init__(self):
        self.calc = SimpleRasterCalc()
        self.salt = random.randint(1, 100000)

    @staticmethod
    def name():
        pass

    @staticmethod
    def description():
        pass

    def init(self, ctx: QgisExecutionContext):
        pass

    def process(self, ctx: QgisExecutionContext) -> str:
        ctx.log_info(f"[{self.name()}]: initializing.")
        self.init(ctx)

        ctx.log_info(f"[{self.name()}]: starting per-band correction.")
        result_bands = self._process_parallel(ctx) if ctx.run_parallel else self._process_sequentially(ctx)

        ctx.log_info(f"[{self.name()}]: merging corrected bands.")

        out_type = get_raster_type(ctx.input_layer_path)
        out_path = ctx.merge_bands(result_bands, out_type)

        ctx.log_info(f"[{self.name()}]: finished.")
        return out_path

    def _process_band_with_metrics(self, ctx: QgisExecutionContext, band_idx: int) -> str:
        start_ns = time.process_time_ns()
        result = self._process_band(ctx, band_idx)
        end_ns = time.process_time_ns()

        ctx.log_info(f"[{self.name()}]: band {band_idx} was processed in {(end_ns - start_ns) / 1000000} ms")
        return result

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int) -> str:
        pass

    def _process_parallel(self, ctx: QgisExecutionContext):
        result_bands = []
        result_futures = []
        serializable_ctx = SerializableQgisExecutionContext.from_ctx(ctx)

        with ProcessPoolExecutor(max_workers=ctx.worker_count) as executor:
            for band_idx in range(ctx.input_layer_band_count):
                future = executor.submit(_process_band_wrapper, self, serializable_ctx, band_idx)
                result_futures.append(future)

                if ctx.is_canceled():
                    [future.cancel() for future in result_futures]
                    executor.shutdown(cancel_futures=True)
                    ctx.force_cancel()

                result_bands.append(self._output_file_path(ctx, str(band_idx)))

            for band_idx, future in enumerate(result_futures):
                try:
                    future.result(timeout=ctx.task_timeout)
                except Exception as exc:
                    ctx.log_error(f"Error during processing band {band_idx}: {traceback.format_exc()}", fatal=True)
                    ctx.force_cancel(exc)

        return result_bands

    def _process_sequentially(self, ctx: QgisExecutionContext):
        result_bands = []

        for band_idx in range(ctx.input_layer_band_count):
            try:
                result = self._process_band_with_metrics(ctx, band_idx)
                result_bands.append(result)
            except Exception as exc:
                ctx.log_error(f"Error during processing band {band_idx}: {traceback.format_exc()}", fatal=True)
                ctx.force_cancel(exc)

            if ctx.is_canceled():
                ctx.force_cancel()

        return result_bands

    def _output_file_path(self, ctx, postfix=''):
        return os.path.join(
            ctx.tmp_dir,
            f'{self.name()}_{self.salt}_{postfix}.tif'
        )

    def _raster_calculate(self, ctx: QgisExecutionContext, calc_func, raster_infos: list[RasterInfo],
                          out_file_postfix='',
                          **kwargs):
        if ctx.is_canceled():
            ctx.force_cancel()

        out_path = self._output_file_path(ctx, out_file_postfix)
        self.calc.calculate(
            func=calc_func,
            output_path=out_path,
            raster_infos=raster_infos,
            **kwargs
        )
        return out_path


def _process_band_wrapper(algorithm, _ctx, _band_idx):
    algorithm._process_band_with_metrics(_ctx, _band_idx)
