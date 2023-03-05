from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from computation import gdal_utils
from processing_alg.topocorrection_eval.eval_algorithm import TopoCorrectionEvalAlgorithm, draw_subplots


def plot_histograms(histograms, luminance_bytes, cmap="seismic", subplots_in_row=4, output_file_path=None, show_plot=True):
    def subplot_histogram(ax, idx):
        histogram, xmin, xmax, ymin, ymax, intercept, slope = histograms[idx]
        img = ax.imshow(
            histogram,
            # interpolation='nearest',
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            aspect=1 / (2 * ymax),
            cmap=cmap
        )
        ax.plot(luminance_bytes, slope * luminance_bytes + intercept, color='red', linewidth=0.5)

        # img = ax.scatter_density(x, y, cmap=cmap)
        plt.colorbar(img, ax=ax, cmap=cmap)

    draw_subplots(
        histograms,
        subplot_histogram,
        subplots_in_row,
        output_file_path=output_file_path,
        show=show_plot,
        figsize=(28, 12),
        # subplot_kw={'projection': 'scatter_density'}
    )


def build_densities(luminance_bytes, img_ds, bins=100, cmap="seismic", output_file_path=None, show_plot=True):
    # luminance = cos(i)
    x_min, x_max = 0, 1

    histograms = []
    for band_idx in range(img_ds.RasterCount):
        band = img_ds.GetRasterBand(band_idx + 1)
        band_bytes = band.ReadAsArray().ravel()
        band_stats = band.GetStatistics(True, True)
        histogram, _, _ = np.histogram2d(
            luminance_bytes,
            band_bytes,
            bins=bins,
            range=[[x_min, x_max], [band_stats[0], band_stats[1]]]
        )

        intercept, slope = np.polynomial.polynomial.polyfit(luminance_bytes, band_bytes, 1)
        histograms.append((histogram.T, x_min, x_max, band_stats[0], band_stats[1], intercept, slope))

    plot_histograms(histograms, luminance_bytes, cmap, output_file_path=output_file_path, show_plot=show_plot)


class RegressionEvalAlgorithm(TopoCorrectionEvalAlgorithm):

    @staticmethod
    def get_name():
        return "Correlation between corrected image and luminance"

    def process_internal(self) -> Any:
        luminance_path = self.ctx.calculate_luminance()
        if self.ctx.qgis_feedback.isCanceled():
            return {}

        luminance_bytes = gdal_utils.read_band_as_array(luminance_path).ravel()
        img_ds = gdal_utils.open_img(self.ctx.input_layer.source())
        build_densities(
            luminance_bytes,
            img_ds,
            bins=100,
            # todo add this as processing algo args
            cmap="coolwarm",
            output_file_path=self.ctx.output_file_path,
            show_plot=False
        )


# luminance_bytes = gdal_utils.read_band_as_array(r"..\..\test\resources\LUMINANCE_3.tif").ravel()
# img_ds = gdal_utils.open_img(r"..\..\test\resources\CUT_INPUT_3.tif")
# build_densities(luminance_bytes, img_ds, cmap="coolwarm")
