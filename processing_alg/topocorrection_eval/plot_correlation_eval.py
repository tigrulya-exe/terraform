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
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from qgis.core import QgsProcessingParameterBoolean, \
    QgsProcessingParameterString, QgsProcessingParameterEnum

from .OutputFormatMixin import OutputFormatMixin
from .correlation_eval import CorrelationEvaluationProcessingAlgorithm, CorrelationEvaluationAlgorithm, \
    CorrelationNodeInfo
from .eval import SubplotMergeStrategy, PlotPerFileMergeStrategy
from ..execution_context import QgisExecutionContext
from ...util.plot_utils import norm_from_scale


def draw_subplot(
        plot_info: CorrelationNodeInfo,
        ax,
        cmap="seismic",
        norm_method="linear",
        plot_regression_line=True):

    xmin, xmax = 0, 1
    ymin, ymax = plot_info.img_stats
    intercept, slope = plot_info.fit_stats

    ax.set_title(plot_info.name)

    plot_info.histogram[plot_info.histogram < 1] = np.NaN
    cm = plt.get_cmap(cmap)
    cm.set_bad("white")

    img = ax.imshow(
        plot_info.histogram,
        norm=norm_from_scale(norm_method),
        # interpolation='nearest',
        origin='lower',
        extent=[xmin, xmax, ymin, ymax],
        aspect=1 / (2 * (ymax - ymin)),
        cmap=cm
    )
    if plot_regression_line:
        ax.plot(plot_info.x_bytes, slope * plot_info.x_bytes + intercept, color='red', linewidth=0.5)
        ax.set_xlabel('cos(i)')
        ax.set_ylabel('radiance')

    plt.colorbar(img, ax=ax, cmap=cm, fraction=0.02874, pad=0.04)


class CorrelationPlotMergeStrategy(SubplotMergeStrategy):
    def __init__(
            self,
            norm_method,
            cmap,
            plot_regression_line,
            subplots_in_row=4,
            path_provider=None,
            figsize=(28, 12),
            subplot_kw=None,
            figname=None):
        super().__init__(subplots_in_row, path_provider, figsize, subplot_kw)
        self.norm_method = norm_method
        self.cmap = cmap
        self.plot_regression_line = plot_regression_line
        self.figname = figname

    def draw_subplot(self, subplot_info, ax, idx):
        draw_subplot(
            subplot_info,
            ax,
            self.cmap,
            self.norm_method,
            self.plot_regression_line
        )

    def after_plot(self, fig, axes):
        fig.suptitle(self.figname, fontsize=16)


class CorrelationPlotPerFileMergeStrategy(PlotPerFileMergeStrategy):
    def __init__(
            self,
            output_directory: str,
            path_provider,
            norm_method="linear",
            cmap="seismic",
            plot_regression_line=True,
            figsize=(28, 12),
            subplot_kw=None):
        super().__init__(output_directory, path_provider, figsize, subplot_kw)
        self.norm_method = norm_method
        self.cmap = cmap
        self.plot_regression_line = plot_regression_line

    def draw_plot(self, plot_info):
        fig, ax = plt.subplots()
        draw_subplot(
            plot_info,
            ax,
            self.cmap,
            self.norm_method,
            self.plot_regression_line
        )


class PlotCorrelationEvaluationProcessingAlgorithm(CorrelationEvaluationProcessingAlgorithm, OutputFormatMixin):
    class ScaleMethod(str, Enum):
        LINEAR = 'linear'
        SYMMETRIC_LOG = 'symlog'

    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)

        self.addParameter(
            QgsProcessingParameterBoolean(
                'DRAW_REGRESSION_LINE',
                self.tr('Draw correlation line on plots'),
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                'DRAW_PER_FILE',
                self.tr('Save band plots in separate file'),
                defaultValue=False
            )
        )

        plot_colormap_param = QgsProcessingParameterString(
            'PLOT_COLORMAP',
            self.tr('Plot colormap'),
            defaultValue='BuPu'
        )
        self._additional_param(plot_colormap_param)

        pixel_scale_param = QgsProcessingParameterEnum(
            'PIXEL_SCALE_METHOD',
            self.tr('The normalization method used to scale pixel values to the [0, 1] range of colormap'),
            options=[e for e in self.ScaleMethod],
            allowMultiple=False,
            defaultValue='symlog',
            usesStaticStrings=True
        )
        self._additional_param(pixel_scale_param)

        pixel_scale_param = self.output_format_param()
        self._additional_param(pixel_scale_param)

    def createInstance(self):
        return PlotCorrelationEvaluationProcessingAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'luminance_radiance_correlation_plot_eval'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Correlation plot between luminance and radiance')

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return super().shortHelpString() + self.tr(
            '\nAlso see Matplotlib docs of <a href="https://matplotlib.org/stable/tutorials/colors/colormapnorms.html">the normalization methods</a> '
            'and <a href="https://matplotlib.org/stable/gallery/color/colormap_reference.html">the colormaps</a> '
            'for additional info about algorithm arguments.')

    def add_layers_to_project(self, ctx: QgisExecutionContext, results):
        if self.format_param_supported_by_qgis(ctx):
            super().add_layers_to_project(ctx, results)

    def _compute_correlation(self, ctx: QgisExecutionContext, group_ids_path):

        output_directory = ctx.output_file_path
        bins = self.parameterAsInt(ctx.qgis_params, 'BIN_COUNT', ctx.qgis_context)
        norm_method = self.parameterAsEnumString(ctx.qgis_params, 'PIXEL_SCALE_METHOD', ctx.qgis_context)
        cmap = self.parameterAsString(ctx.qgis_params, 'PLOT_COLORMAP', ctx.qgis_context)
        plot_regression_line = self.parameterAsBoolean(ctx.qgis_params, 'DRAW_REGRESSION_LINE', ctx.qgis_context)

        per_file = self.parameterAsBoolean(ctx.qgis_params, 'DRAW_PER_FILE', ctx.qgis_context)
        output_format = self.get_output_format_param(ctx)

        if per_file:
            def generate_file_name(node: CorrelationNodeInfo):
                return f"{ctx.input_file_name}_correlation_{node.group_idx}_{node.name}.{output_format}"

            merge_strategy = CorrelationPlotPerFileMergeStrategy(
                output_directory,
                generate_file_name,
                norm_method,
                cmap,
                plot_regression_line
            )
        else:
            def generate_file_name(nodes: list[CorrelationNodeInfo]):
                filename = f"{ctx.input_file_name}_plot_correlation_{nodes[0].group_idx}.{output_format}"
                return os.path.join(output_directory, filename)

            merge_strategy = CorrelationPlotMergeStrategy(
                norm_method,
                cmap,
                plot_regression_line,
                path_provider=generate_file_name,
                figname=ctx.input_file_name
            )

        alg = CorrelationEvaluationAlgorithm(
            ctx,
            merge_strategy,
            ctx.luminance_path,
            bins,
            group_ids_path,
        )

        output_paths = alg.evaluate()

        return [(path, os.path.basename(path)) for path in output_paths]
