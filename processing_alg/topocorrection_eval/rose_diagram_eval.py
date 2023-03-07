import math
from typing import Any, Dict

import numpy as np
import numpy_groupies as npg
from matplotlib import pyplot as plt
from qgis._core import QgsProcessingParameterNumber
from qgis.core import QgsProcessingFeedback

from computation import gdal_utils
from processing_alg.execution_context import QgisExecutionContext
from processing_alg.topocorrection_eval.topocorrection_eval_algorithm import TopocorrectionEvaluationAlgorithm

MPLT_MARKERS = "ov^<>1235sp*X"
MPLT_COLOURS = "bgrcmyk"


def divide_to_groups(groups_count, upper_bound, lower_bound=0):
    group_size = (upper_bound - lower_bound) // groups_count
    return [lower_bound + i * group_size for i in range(groups_count)]


def get_slope_label(slope_groups_bounds, idx):
    higher_bound = "+" if len(slope_groups_bounds) == idx + 1 else f"-{slope_groups_bounds[idx + 1]}"
    return f"{slope_groups_bounds[idx]}{higher_bound}°"


def plot_rose_diagrams(
        group_means_list,
        slope_groups_count=3,
        slope_max_deg=90.0,
        aspect_groups_count=36,
        aspect_max_deg=360.0,
        subplots_in_row=4,
        # todo replace with enum
        output_file_path=None,
        show=True):
    slope_groups_bounds = divide_to_groups(slope_groups_count, upper_bound=slope_max_deg)
    aspect_groups_bounds = divide_to_groups(aspect_groups_count, upper_bound=aspect_max_deg)

    non_empty_aspect_groups = group_means_list[0].shape[1]
    aspect_bounds_rad = [math.radians(deg) for deg in aspect_groups_bounds[:non_empty_aspect_groups]]

    row_count = len(group_means_list) // subplots_in_row
    if len(group_means_list) % subplots_in_row != 0:
        row_count += 1

    fig, axes = plt.subplots(row_count, subplots_in_row, subplot_kw=dict(projection='polar'), figsize=(15, 15))

    for idx, ax in enumerate(axes.flat):
        if idx >= len(group_means_list):
            fig.delaxes(ax)
            continue

        group_means = group_means_list[idx]
        # tick - 30 degree
        ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
        ax.set_rlabel_position(0)
        ax.set_title(f'Band №{idx + 1}')

        for slope_bound_idx, subgroup_means in enumerate(group_means):
            point_design = MPLT_COLOURS[(slope_bound_idx * 2) % len(MPLT_COLOURS)] + \
                           MPLT_MARKERS[slope_bound_idx % len(MPLT_MARKERS)]
            ax.plot(aspect_bounds_rad, subgroup_means, point_design,
                    label=get_slope_label(slope_groups_bounds, slope_bound_idx))

        # ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, title='Slope')
        ax.tick_params(axis='y', rotation=45)

    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        borderaxespad=0,
        title='Slope',
        fontsize=16
    )

    plt.tight_layout()
    if output_file_path is not None:
        plt.savefig(output_file_path, bbox_inches="tight")
    if show:
        plt.show()


def group_by_range(arr, groups_count, upper_bound, lower_bound=0):
    group_size = (upper_bound - lower_bound) // groups_count
    return ((arr - lower_bound) // group_size).astype(int, copy=False)


def calculate_rose_groups(
        img_bytes,
        slope_bytes,
        aspect_bytes,
        slope_groups_count=3,
        slope_max_deg=90,
        aspect_groups_count=36,
        aspect_max_deg=360):
    slope_groups = group_by_range(slope_bytes, slope_groups_count, upper_bound=slope_max_deg)
    aspect_groups = group_by_range(aspect_bytes, aspect_groups_count, upper_bound=aspect_max_deg)
    groups = np.vstack((slope_groups, aspect_groups))

    group_means = npg.aggregate(groups, img_bytes, func='mean', fill_value=0)
    return group_means


def get_flat_band(ds, band_idx):
    return ds.GetRasterBand(band_idx).ReadAsArray().ravel()


def build_polar_diagrams(
        img_path,
        slope_path,
        aspect_path,
        slope_groups_count=3,
        slope_max_deg=90.0,
        aspect_groups_count=36,
        aspect_max_deg=360.0,
        output_file_path=None,
        show_plot=True):
    slope_bytes = gdal_utils.read_band_as_array(slope_path).ravel()
    aspect_bytes = gdal_utils.read_band_as_array(aspect_path).ravel()

    img_ds = gdal_utils.open_img(img_path)

    slope_groups = group_by_range(slope_bytes, slope_groups_count, upper_bound=slope_max_deg)
    aspect_groups = group_by_range(aspect_bytes, aspect_groups_count, upper_bound=aspect_max_deg)
    groups_idxs = np.vstack((slope_groups, aspect_groups))

    groups = []
    for band_idx in range(img_ds.RasterCount):
        band_bytes = get_flat_band(img_ds, band_idx + 1)
        group_means = npg.aggregate(groups_idxs, band_bytes, func='mean', fill_value=0)
        groups.append(group_means)

    plot_rose_diagrams(groups,
                       slope_groups_count=slope_groups_count,
                       slope_max_deg=slope_max_deg,
                       aspect_groups_count=aspect_groups_count,
                       aspect_max_deg=aspect_max_deg,
                       output_file_path=output_file_path,
                       show=show_plot)
    return groups


class RoseDiagramEvaluationAlgorithm(TopocorrectionEvaluationAlgorithm):

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
            QgsProcessingParameterNumber(
                'SLOPE_MAX_DEGREE',
                self.tr('Slope degrees top limit'),
                defaultValue=90.0,
                type=QgsProcessingParameterNumber.Double,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'ASPECT_MAX_DEGREE',
                self.tr('Aspect degrees top limit'),
                defaultValue=360.0,
                type=QgsProcessingParameterNumber.Double,
                optional=True
            )
        )

    def createInstance(self):
        return RoseDiagramEvaluationAlgorithm()

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
        return self.tr('Contains rose diagrams topocorrection algorithm evaluation method.')

    def processAlgorithmInternal(
            self,
            parameters: Dict[str, Any],
            context: QgisExecutionContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        slope_path = context.calculate_slope(in_radians=False)
        if context.qgis_feedback.isCanceled():
            return {}

        aspect_path = context.calculate_aspect(in_radians=False)
        if context.qgis_feedback.isCanceled():
            return {}

        build_polar_diagrams(
            context.input_layer.source(),
            slope_path,
            aspect_path,
            slope_groups_count=self.parameterAsInt(parameters, 'SLOPE_GROUPS_COUNT', context.qgis_context),
            slope_max_deg=self.parameterAsDouble(parameters, 'SLOPE_MAX_DEGREE', context.qgis_context),
            aspect_groups_count=self.parameterAsInt(parameters, 'ASPECT_GROUPS_COUNT', context.qgis_context),
            aspect_max_deg=self.parameterAsDouble(parameters, 'ASPECT_MAX_DEGREE', context.qgis_context),
            output_file_path=context.output_file_path,
            show_plot=False
        )

        return {}
