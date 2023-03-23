import os
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from ...computation import gdal_utils
from ...processing_alg.execution_context import QgisExecutionContext


class MergeStrategy:
    def merge(self, results):
        pass


class PerFileMergeStrategy(MergeStrategy):

    def __init__(self, output_directory: str, path_provider) -> None:
        self.output_directory = output_directory
        self.filename_provider = path_provider

    def merge(self, results):
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
            output_file_path=None,
            figsize=(15, 15),
            subplot_kw=None):
        self.subplots_in_row = subplots_in_row
        self.output_file_path = output_file_path
        self.figsize = figsize
        self.subplot_kw = subplot_kw

    def merge(self, subplot_infos):
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

        plt.tight_layout()
        if self.output_file_path is not None:
            plt.savefig(self.output_file_path, bbox_inches="tight")

        # todo tmp
        return [self.output_file_path]

    def after_plot(self, fig, axes):
        pass

    def draw_subplot(self, subplot_info, ax, idx):
        pass


class EvaluationAlgorithm:
    def __init__(
            self,
            ctx: QgisExecutionContext,
            merge_strategy: MergeStrategy,
            group_ids_path=None):
        self.ctx = ctx
        self.merge_strategy = merge_strategy

        self.input_ds = gdal_utils.open_img(ctx.input_layer.source())

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

        for group in self.groups:
            band_results = []
            for band_idx in range(self.input_ds.RasterCount):
                band = self.input_ds.GetRasterBand(band_idx + 1)

                band_bytes = band.ReadAsArray().ravel()[self.groups_map == group]

                band_result = self.evaluate_band(band_bytes, band_idx, band, group)
                band_results.append(band_result)

                if self.ctx.is_canceled():
                    return None

            result += self.merge_strategy.merge(band_results)
        return result

    def evaluate_band(self, band_bytes, gdal_band, band_idx, group_idx) -> Any:
        pass
