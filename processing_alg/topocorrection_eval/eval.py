import os
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from ...processing_alg.execution_context import QgisExecutionContext
from ...util import gdal_utils


class MergeStrategy:
    def merge(self, results, group_idx):
        pass


class PerFileMergeStrategy(MergeStrategy):

    def __init__(self, output_directory: str, path_provider) -> None:
        self.output_directory = output_directory
        self.filename_provider = path_provider

    def merge(self, results, group_idx):
        result_paths = []
        for result in results:
            filename = self.filename_provider(result)
            result_paths.append(os.path.join(self.output_directory, filename))
            self.save_result(result, filename)
        return result_paths

    def save_result(self, result, filename):
        pass


class PlotPerFileMergeStrategy(PerFileMergeStrategy):
    def __init__(self, output_directory: str, path_provider, figsize=(15, 15), subplot_kw=None):
        super().__init__(output_directory, path_provider)
        self.figsize = figsize
        self.subplot_kw = subplot_kw

    def save_result(self, subplot_info, filename):
        plot_path = os.path.join(self.output_directory, filename)
        self.draw_plot(subplot_info)
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")

    def draw_plot(self, plot_info):
        pass


class SubplotMergeStrategy(MergeStrategy):
    def __init__(
            self,
            subplots_in_row=4,
            path_provider=None,
            figsize=(15, 15),
            subplot_kw=None):
        self.subplots_in_row = subplots_in_row
        self.path_provider = path_provider or SubplotMergeStrategy._default_path_provider
        self.figsize = figsize
        self.subplot_kw = subplot_kw

    @staticmethod
    def _default_path_provider(_):
        return None

    def merge(self, subplot_infos, group_idx):
        row_count = len(subplot_infos) // self.subplots_in_row
        if len(subplot_infos) % self.subplots_in_row != 0:
            row_count += 1

        fig, axes = plt.subplots(row_count, self.subplots_in_row, figsize=self.figsize, subplot_kw=self.subplot_kw)
        for idx, ax in enumerate(axes.flat):
            if idx >= len(subplot_infos):
                fig.delaxes(ax)
                continue

            self.draw_subplot(subplot_infos[idx], ax, idx)

        self.after_plot(fig, axes)

        output_file_path = self.path_provider(subplot_infos)
        plt.tight_layout()
        if output_file_path is not None:
            plt.savefig(output_file_path, bbox_inches="tight")

        return [output_file_path]

    def after_plot(self, fig, axes):
        pass

    def draw_subplot(self, subplot_info, ax, idx):
        pass


class EvaluationAlgorithm:
    @dataclass
    class BandInfo:
        def __init__(self, gdal_band, band_bytes, idx):
            self.gdal_band = gdal_band
            self.bytes = band_bytes
            self.idx = idx

    def __init__(
            self,
            ctx: QgisExecutionContext,
            merge_strategy: MergeStrategy,
            group_ids_path=None):
        self.ctx = ctx
        self.merge_strategy = merge_strategy

        self.input_ds = gdal_utils.open_img(ctx.input_layer_path)

        if group_ids_path is not None:
            self.groups_map = gdal_utils.read_band_as_array(group_ids_path, 1).ravel()
            groups = np.unique(self.groups_map)
            self.groups = groups[~np.isnan(groups)]
        else:
            band = self.input_ds.GetRasterBand(1)
            self.groups_map = np.full(band.YSize * band.XSize, 1, dtype=int)
            self.groups = [1]

    def evaluate(self):
        result = []

        try:
            for group in self.groups:
                self.ctx.log_info(f"Start evaluating group {group}.")
                result += self._evaluate_group(group)
                self.ctx.log_info(f"Group {group} evaluated.")
        except Exception as exc:
            self.ctx.log_error(f"Error during evaluation: {traceback.format_exc()}", fatal=True)
            self.ctx.force_cancel(exc)

        return result

    def _evaluate_group(self, group):
        band_results = self._evaluate_raster(self.input_ds, group)
        return self.merge_strategy.merge(band_results, group)

    def _evaluate_raster(self, raster_ds, group_idx):
        band_results = []
        for band_idx in range(raster_ds.RasterCount):
            original_band = self._get_masked_band(raster_ds, band_idx, group_idx)

            band_result = self._evaluate_band(original_band, group_idx)
            band_results.append(band_result)

            if self.ctx.is_canceled():
                self.ctx.force_cancel()
        return band_results

    def _get_masked_band(self, ds, band_idx, group_idx) -> BandInfo:
        orig_band = ds.GetRasterBand(band_idx + 1)
        orig_band_bytes = self._get_masked_bytes(orig_band.ReadAsArray().ravel(), group_idx)
        return self.BandInfo(orig_band, orig_band_bytes, band_idx)

    def _get_masked_bytes(self, band_bytes, group_idx):
        return band_bytes[self.groups_map == group_idx]

    def _evaluate_band(self, band: BandInfo, group_idx) -> Any:
        pass
