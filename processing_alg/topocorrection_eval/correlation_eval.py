from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from qgis._core import QgsProcessingFeedback, QgsProcessingParameterNumber, QgsProcessingParameterBoolean, \
    QgsProcessingParameterString

from computation import gdal_utils
from computation.plot_utils import draw_subplots
from processing_alg.execution_context import QgisExecutionContext
from processing_alg.topocorrection_eval.topocorrection_eval_algorithm import TopocorrectionEvaluationAlgorithm


def plot_histograms(
        histograms,
        luminance_bytes,
        cmap="seismic",
        plot_regression_line=True,
        subplots_in_row=4,
        output_file_path=None,
        show_plot=True):
    def subplot_histogram(ax, idx):
        histogram, xmin, xmax, ymin, ymax, intercept, slope = histograms[idx]
        img = ax.imshow(
            histogram,
            # interpolation='nearest',
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            aspect=1 / (2 * (ymax - ymin)),
            cmap=cmap
        )
        if plot_regression_line:
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


def build_densities(
        luminance_bytes,
        img_ds,
        bins=100,
        cmap="seismic",
        plot_regression_line=True,
        output_file_path=None,
        show_plot=True):
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

    plot_histograms(
        histograms,
        luminance_bytes,
        cmap,
        output_file_path=output_file_path,
        plot_regression_line=plot_regression_line,
        show_plot=show_plot)


# luminance_bytes = gdal_utils.read_band_as_array(r"..\..\test\resources\LUMINANCE_3.tif").ravel()
# img_ds = gdal_utils.open_img(r"..\..\test\resources\CUT_INPUT_3.tif")
# build_densities(luminance_bytes, img_ds, cmap="coolwarm")

class CorrelationEvaluationAlgorithm(TopocorrectionEvaluationAlgorithm):
    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)

        self.addParameter(
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Solar zenith angle'),
                defaultValue=57.2478878065826,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar azimuth'),
                defaultValue=177.744663052425,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                'DRAW_REGRESSION_LINE',
                self.tr('Draw regression line'),
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'BIN_COUNT',
                self.tr('Number of bins per axis in 2d diagram'),
                defaultValue=100,
                type=QgsProcessingParameterNumber.Integer,
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'PLOT_COLORMAP',
                self.tr('Plot colormap'),
                defaultValue='coolwarm'
            )
        )

    def createInstance(self):
        return CorrelationEvaluationAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'luminance_radiance_correlation_eval'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Correlation between luminance and radiance')

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Computes correlation between corrected image and luminance.')

    def processAlgorithmInternal(
            self,
            parameters: Dict[str, Any],
            context: QgisExecutionContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        context.sza_degrees = self.parameterAsDouble(parameters, 'SZA', context.qgis_context)
        context.solar_azimuth_degrees = self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context.qgis_context)

        luminance_path = context.calculate_luminance()
        if context.qgis_feedback.isCanceled():
            return {}

        luminance_bytes = gdal_utils.read_band_as_array(luminance_path).ravel()
        img_ds = gdal_utils.open_img(context.input_layer.source())
        build_densities(
            luminance_bytes,
            img_ds,
            bins=self.parameterAsInt(parameters, 'BIN_COUNT', context.qgis_context),
            cmap=self.parameterAsString(parameters, 'PLOT_COLORMAP', context.qgis_context),
            plot_regression_line=self.parameterAsBoolean(parameters, 'DRAW_REGRESSION_LINE', context.qgis_context),
            output_file_path=context.output_file_path,
            show_plot=False
        )

        return {}
