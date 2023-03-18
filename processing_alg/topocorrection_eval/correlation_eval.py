import os
from enum import Enum
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from qgis.core import QgsProcessingParameterRasterLayer, \
    QgsProcessingFeedback, QgsProcessingParameterNumber, QgsProcessingParameterBoolean, \
    QgsProcessingParameterString, QgsProcessingParameterEnum

from .qgis_algorithm import TopocorrectionEvaluationAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation import gdal_utils
from ...computation.plot_utils import draw_subplots, norm_from_scale
from ...computation.qgis_utils import check_compatible, set_layers_to_load


def plot_histograms(
        histograms,
        cmap="seismic",
        norm_method="linear",
        plot_regression_line=True,
        subplots_in_row=4,
        output_file_path=None,
        show_plot=True):
    def subplot_histogram(ax, idx):
        histogram, luminance_bytes, xmin, xmax, ymin, ymax, intercept, slope = histograms[idx]

        ax.set_title(f'Band №{idx + 1}')
        img = ax.imshow(
            histogram,
            norm=norm_from_scale(norm_method),
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
        norm_method="linear",
        group_ids=None,
        plot_regression_line=True,
        output_file_path=None,
        show_plot=True):
    # luminance = cos(i)
    x_min, x_max = 0, 1

    groups = np.unique(group_ids)
    groups = groups[~np.isnan(groups)]

    output_paths = []
    # memory/speed tradeoff
    for group in groups:
        histograms = []
        for band_idx in range(img_ds.RasterCount):
            band = img_ds.GetRasterBand(band_idx + 1)
            band_bytes = band.ReadAsArray().ravel()[group_ids == group]
            group_luminance_bytes = luminance_bytes[group_ids == group]

            # todo change to band.minmax()
            img_min, img_max, *_ = band.GetStatistics(True, True)
            histogram, _, _ = np.histogram2d(
                group_luminance_bytes,
                band_bytes,
                bins=bins,
                range=[[x_min, x_max], [img_min, img_max]]
            )

            intercept, slope = np.polynomial.polynomial.polyfit(group_luminance_bytes, band_bytes, 1)
            histograms.append((histogram.T, group_luminance_bytes, x_min, x_max, img_min, img_max, intercept, slope))

        if len(groups) == 1:
            group_out_path = output_file_path
        else:
            path_prefix, path_ext = os.path.splitext(output_file_path)
            group_out_path = f"{path_prefix}_{group}{path_ext}"
            output_paths.append(group_out_path)

        plot_histograms(
            histograms,
            cmap,
            norm_method=norm_method,
            output_file_path=group_out_path,
            plot_regression_line=plot_regression_line,
            show_plot=show_plot)

    return output_paths


class CorrelationEvaluationAlgorithm(TopocorrectionEvaluationAlgorithm):
    class ScaleMethod(str, Enum):
        LINEAR = 'linear'
        SYMMETRIC_LOG = 'symlog'

    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)

        self.addParameter(
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Solar zenith angle (in degrees)'),
                defaultValue=57.2478878065826,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar azimuth (in degrees)'),
                defaultValue=177.744663052425,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                'DRAW_REGRESSION_LINE',
                self.tr('Draw correlation line on plots'),
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

        plot_colormap_param = QgsProcessingParameterString(
            'PLOT_COLORMAP',
            self.tr('Plot colormap'),
            defaultValue='coolwarm'
        )
        self._additional_param(plot_colormap_param)

        pixel_scale_param = QgsProcessingParameterEnum(
            'PIXEL_SCALE_METHOD',
            self.tr('The normalization method used to scale pixel values to the [0, 1] range of colormap'),
            options=[e for e in self.ScaleMethod],
            allowMultiple=False,
            defaultValue='linear',
            usesStaticStrings=True
        )
        self._additional_param(pixel_scale_param)

        classification_map_param = QgsProcessingParameterRasterLayer(
            'CLASSIFICATION_MAP',
            self.tr('Raster layer with classification label ids for input raster'),
            optional=True
        )
        self._additional_param(classification_map_param)

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
        return self.tr('Builds density plots (2d histograms) of the relationship between '
                       'raster image bands and illumination model of that image.\n'
                       'Also see Matplotlib docs of <a href="https://matplotlib.org/stable/tutorials/colors/colormapnorms.html">the normalization methods</a> '
                       'and <a href="https://matplotlib.org/stable/gallery/color/colormap_reference.html">the colormaps</a> '
                       'for additional info about algorithm arguments.\n'
                       "<b>Note:</b> the illumination model of the input raster image is calculated automatically, "
                       "based on the provided DEM layer. Currently, the input raster image and the DEM must have "
                       "the same CRS, extent and spatial resolution.")

    def need_to_show_result(self, execution_ctx: QgisExecutionContext):
        # In this algorithm we manually set output file(s)
        return False

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

        group_ids_layer = self.parameterAsRasterLayer(parameters, 'CLASSIFICATION_MAP', context.qgis_context)
        if group_ids_layer is not None:
            check_compatible(group_ids_layer, context.input_layer)
            # todo read as dtype=int
            group_ids_bytes = gdal_utils.read_band_as_array(luminance_path).ravel()
        else:
            group_ids_bytes = np.full_like(luminance_bytes, 1, dtype=int)

        output_paths = build_densities(
            luminance_bytes,
            img_ds,
            bins=self.parameterAsInt(parameters, 'BIN_COUNT', context.qgis_context),
            cmap=self.parameterAsString(parameters, 'PLOT_COLORMAP', context.qgis_context),
            plot_regression_line=self.parameterAsBoolean(parameters, 'DRAW_REGRESSION_LINE', context.qgis_context),
            norm_method=self.parameterAsEnumString(parameters, 'PIXEL_SCALE_METHOD', context.qgis_context),
            group_ids=group_ids_bytes,
            output_file_path=context.output_file_path,
            show_plot=False
        )

        paths_with_names = [(path, os.path.basename(path)) for path in output_paths]
        set_layers_to_load(context.qgis_context, paths_with_names)

        return {}
