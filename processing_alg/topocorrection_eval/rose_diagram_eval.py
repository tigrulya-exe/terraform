import math
import os
from typing import Any, Dict

import numpy as np
import numpy_groupies as npg
from matplotlib import pyplot as plt
from qgis.core import QgsProcessingFeedback, QgsProcessingParameterNumber, QgsProcessingParameterBoolean, \
    QgsProcessingParameterRasterLayer

from .OutputFormatMixin import OutputFormatMixin
from .eval import MergeStrategy, EvaluationAlgorithm, SubplotMergeStrategy, PlotPerFileMergeStrategy
from .qgis_algorithm import TopocorrectionEvaluationAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation import gdal_utils

MPLT_MARKERS = "ov^<>1235sp*X"
MPLT_COLOURS = "bgrcmyk"


def divide_to_groups(groups_count, upper_bound, lower_bound=0):
    group_size = (upper_bound - lower_bound) // groups_count
    return [lower_bound + i * group_size for i in range(groups_count)]


def get_slope_label(slope_groups_bounds, idx):
    higher_bound = "+" if len(slope_groups_bounds) == idx + 1 else f"-{slope_groups_bounds[idx + 1]}"
    return f"{slope_groups_bounds[idx]}{higher_bound}°"


def compute_statistics(array) -> dict[str, float]:
    percentiles = [0, 50, 95, 97, 99]
    percentiles_values = {
        f'percentile_{percentiles[idx]}': value for idx, value in enumerate(np.percentile(array, percentiles))
    }
    return {
               'mean': np.mean(array),
               'stddev': np.std(array),
           } | percentiles_values


class RoseDiagramsNodeInfo:
    def __init__(
            self,
            group_means,
            name,
            group_idx=None):
        self.group_means = group_means
        self.name = name
        self.group_idx = group_idx


def _draw_subplot(
        plot_info: RoseDiagramsNodeInfo,
        ax,
        slope_groups_bounds,
        aspect_bounds_rad):
    stats = ",\n".join(
        [f"{name} = {value:.3f}" for name, value in compute_statistics(plot_info.group_means.ravel()).items()])

    # tick - 30 degree
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_rlabel_position(0)
    ax.set_title(plot_info.name)

    for slope_bound_idx, subgroup_means in enumerate(plot_info.group_means):
        point_design = MPLT_COLOURS[(slope_bound_idx * 2) % len(MPLT_COLOURS)] + \
                       MPLT_MARKERS[slope_bound_idx % len(MPLT_MARKERS)]
        ax.plot(aspect_bounds_rad, subgroup_means, point_design,
                label=get_slope_label(slope_groups_bounds, slope_bound_idx))
        ax.text(.5, -.1, stats,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes)

    # ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, title='Slope')
    ax.tick_params(axis='y', rotation=45)


def group_by_range(arr, groups_count, upper_bound, lower_bound=0):
    group_size = (upper_bound - lower_bound) // groups_count
    return ((arr - lower_bound) // group_size).astype(int, copy=False)


def get_flat_band(ds, band_idx):
    return ds.GetRasterBand(band_idx).ReadAsArray().ravel()


class RoseDiagramMergeStrategy(SubplotMergeStrategy):
    def __init__(
            self,
            slope_groups_count=3,
            slope_max_deg=90.0,
            aspect_groups_count=36,
            aspect_max_deg=360.0,
            subplots_in_row=4,
            output_file_path=None,
            figsize=(20, 20)):
        super().__init__(subplots_in_row, output_file_path, figsize, dict(projection='polar'))
        self.slope_groups_bounds = divide_to_groups(slope_groups_count, upper_bound=slope_max_deg)
        self.aspect_groups_bounds = divide_to_groups(aspect_groups_count, upper_bound=aspect_max_deg)
        self.aspect_bounds_rad = None

    def merge(self, subplot_infos):
        non_empty_aspect_groups = subplot_infos[0].group_means.shape[1]
        self.aspect_bounds_rad = [math.radians(deg) for deg in self.aspect_groups_bounds[:non_empty_aspect_groups]]
        return super().merge(subplot_infos)

    def after_plot(self, fig, axes):
        handles, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            borderaxespad=0,
            title='Slope',
            fontsize=16,
            title_fontsize=18
        )

    def draw_subplot(self, subplot_info, ax, idx):
        _draw_subplot(
            subplot_info,
            ax,
            self.slope_groups_bounds,
            self.aspect_bounds_rad
        )


class RoseDiagramPerFileMergeStrategy(PlotPerFileMergeStrategy):
    def __init__(
            self,
            output_directory: str,
            path_provider,
            slope_groups_count=3,
            slope_max_deg=90.0,
            aspect_groups_count=36,
            aspect_max_deg=360.0,
            figsize=(20, 20)):
        super().__init__(output_directory, path_provider, figsize, dict(projection='polar'))
        self.slope_groups_bounds = divide_to_groups(slope_groups_count, upper_bound=slope_max_deg)
        self.aspect_groups_bounds = divide_to_groups(aspect_groups_count, upper_bound=aspect_max_deg)

    def draw_plot(self, plot_info):
        non_empty_aspect_groups = plot_info.group_means.shape[1]
        aspect_bounds_rad = [math.radians(deg) for deg in self.aspect_groups_bounds[:non_empty_aspect_groups]]

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        _draw_subplot(
            plot_info,
            ax,
            self.slope_groups_bounds,
            aspect_bounds_rad
        )

        handles, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            borderaxespad=0,
            title='Slope',
            fontsize=16,
            title_fontsize=18
        )


class RoseDiagramEvaluationAlgorithm(EvaluationAlgorithm):
    def __init__(
            self,
            ctx: QgisExecutionContext,
            merge_strategy: MergeStrategy,
            slope_groups_count=3,
            slope_max_deg=90.0,
            aspect_groups_count=36,
            aspect_max_deg=360.0,
            group_ids=None):
        super().__init__(ctx, merge_strategy, group_ids)

        slope_path = ctx.calculate_slope(in_radians=False)
        aspect_path = ctx.calculate_aspect(in_radians=False)

        slope_bytes = gdal_utils.read_band_as_array(slope_path).ravel()
        aspect_bytes = gdal_utils.read_band_as_array(aspect_path).ravel()

        slope_groups = group_by_range(slope_bytes, slope_groups_count, upper_bound=slope_max_deg)
        aspect_groups = group_by_range(aspect_bytes, aspect_groups_count, upper_bound=aspect_max_deg)
        self.groups_idxs = np.vstack((slope_groups, aspect_groups))

    def evaluate_band(self, band_bytes, band_idx, gdal_band, group_idx) -> Any:
        groups_idxs = self.groups_idxs[:, self.groups_map == group_idx]

        group_means = npg.aggregate(groups_idxs, band_bytes, func='mean', fill_value=0)
        return RoseDiagramsNodeInfo(
            group_means,
            gdal_band.GetDescription(),
            group_idx
        )


class RoseDiagramEvaluationProcessingAlgorithm(TopocorrectionEvaluationAlgorithm, OutputFormatMixin):
    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)

        self.addParameter(
            QgsProcessingParameterNumber(
                'SLOPE_GROUPS_COUNT',
                self.tr('Slope groups count'),
                defaultValue=3,
                type=QgsProcessingParameterNumber.Integer,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'ASPECT_GROUPS_COUNT',
                self.tr('Aspect groups count'),
                defaultValue=36,
                type=QgsProcessingParameterNumber.Integer,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                'DRAW_PER_FILE',
                self.tr('Save band plots in separate file'),
                defaultValue=False
            )
        )

        slope_max_degree_param = QgsProcessingParameterNumber(
            'SLOPE_MAX_DEGREE',
            self.tr('Slope degrees top limit'),
            defaultValue=90.0,
            type=QgsProcessingParameterNumber.Double,
            optional=True
        )
        self._additional_param(slope_max_degree_param)

        aspect_max_degree_param = QgsProcessingParameterNumber(
            'ASPECT_MAX_DEGREE',
            self.tr('Aspect degrees top limit'),
            defaultValue=360.0,
            type=QgsProcessingParameterNumber.Double,
            optional=True
        )
        self._additional_param(aspect_max_degree_param)

        pixel_scale_param = self.output_format_param()
        self._additional_param(pixel_scale_param)

        classification_map_param = QgsProcessingParameterRasterLayer(
            'CLASSIFICATION_MAP',
            self.tr('Raster layer with classification label ids for input raster'),
            optional=True
        )
        self._additional_param(classification_map_param)

    def createInstance(self):
        return RoseDiagramEvaluationProcessingAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'rose_diagram_eval'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Rose diagram evaluation method')

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr("Radiances of a remotely sensed image were divided into the <i>Slope groups count</i> "
                       "groups according to the fixed slope intervals of the terrain "
                       "(the maximum considered slope can be configured by argument <i>Slope degrees top limit</i>). "
                       "For each group, radiances were divided into <i>Aspect groups count</i> "
                       "subgroups according to the fixed aspect intervals of the terrain "
                       "(the maximum considered aspect can be configured by argument <i>Slope degrees top limit</i>). "
                       "Next, mean values of the radiances of the sub-groups were calculated "
                       "and plotted on the rose diagram. The polar angle of the rose diagram "
                       "represents the aspect of terrain, and the radius represents the value "
                       "of radiances. \n"
                       "If the corrected radiances show no azimuth dependence, i.e the subgroup "
                       "radiances’ standard deviation is close to zero, the topographic correction "
                       "can be considered to be successful. Thus, the case of a successful correction "
                       "can be seen both visually (the points on the diagram form a circle) and with "
                       "the help of subsequent statistical analysis of the radiation values of the "
                       "subgroups (small values of standard deviation, coefficient of variation, etc.). "
                       "In addition, using this method, it is relatively easy to determine for surfaces "
                       "with which slopes and aspects a topographic correction algorithm works well, "
                       "and for which it needs to be improved. \n"
                       "<b>Note:</b> the slope and aspect of the input raster image are calculated automatically, "
                       "based on the provided DEM layer. Currently, the input raster image and the DEM must have "
                       "the same CRS, extent and spatial resolution.")

    def add_layers_to_project(self, ctx: QgisExecutionContext, results):
        if self.format_param_supported_by_qgis(ctx):
            super().add_layers_to_project(ctx, results)

    def processAlgorithmInternal(
            self,
            parameters: Dict[str, Any],
            ctx: QgisExecutionContext,
            feedback: QgsProcessingFeedback
    ):
        per_file = self.parameterAsBoolean(ctx.qgis_params, 'DRAW_PER_FILE', ctx.qgis_context)
        output_format = self.get_output_format_param(ctx)

        slope_groups_count = self.parameterAsInt(parameters, 'SLOPE_GROUPS_COUNT', ctx.qgis_context)
        slope_max_deg = self.parameterAsDouble(parameters, 'SLOPE_MAX_DEGREE', ctx.qgis_context)
        aspect_groups_count = self.parameterAsInt(parameters, 'ASPECT_GROUPS_COUNT', ctx.qgis_context)
        aspect_max_deg = self.parameterAsDouble(parameters, 'ASPECT_MAX_DEGREE', ctx.qgis_context)

        group_ids_layer = self.parameterAsRasterLayer(parameters, 'CLASSIFICATION_MAP', ctx.qgis_context)
        group_ids_path = None if group_ids_layer is None else group_ids_layer.source()

        def generate_file_name(node: RoseDiagramsNodeInfo):
            return f"rose_diagram_{node.group_idx}_{node.name}.{output_format}"

        if per_file:
            merge_strategy = RoseDiagramPerFileMergeStrategy(
                ctx.output_file_path,
                generate_file_name,
                slope_groups_count,
                slope_max_deg,
                aspect_groups_count,
                aspect_max_deg,
            )
        else:
            merge_strategy = RoseDiagramMergeStrategy(
                slope_groups_count,
                slope_max_deg,
                aspect_groups_count,
                aspect_max_deg,
                output_file_path=os.path.join(ctx.output_file_path, f"rose_diagram.{output_format}")
            )

        # todo move rose connected input to separate class and instantiate in once
        alg = RoseDiagramEvaluationAlgorithm(
            ctx,
            merge_strategy,
            slope_groups_count,
            slope_max_deg,
            aspect_groups_count,
            aspect_max_deg,
            group_ids_path
        )

        results = alg.evaluate()

        return [(result, os.path.basename(result)) for result in results]
