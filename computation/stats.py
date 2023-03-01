import math

import matplotlib.pyplot as plt
import numpy as np
import numpy_groupies as npg
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

MPLT_MARKERS = "ov^<>1235sp*X"
MPLT_COLOURS = "bgrcmyk"


def divide_to_groups(groups_count, upper_bound, lower_bound=0):
    group_size = (upper_bound - lower_bound) // groups_count
    return [lower_bound + i * group_size for i in range(groups_count)]


def get_slope_label(slope_groups_bounds, idx):
    higher_bound = "+" if len(slope_groups_bounds) == idx + 1 else f"-{slope_groups_bounds[idx + 1]}"
    return f"{slope_groups_bounds[idx]}{higher_bound}"


def plot_rose_diagram(
        group_means,
        slope_groups_count=3,
        slope_max_deg=90,
        aspect_groups_count=36,
        aspect_max_deg=360):
    slope_groups_bounds = divide_to_groups(slope_groups_count, upper_bound=slope_max_deg)
    aspect_groups_bounds = divide_to_groups(aspect_groups_count, upper_bound=aspect_max_deg)

    non_empty_aspect_groups = group_means.shape[1]
    aspect_bounds_rad = [math.radians(deg) for deg in aspect_groups_bounds[:non_empty_aspect_groups]]

    ax = plt.subplot(111, polar=True)
    # tick - 30 degree
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_rlabel_position(0)

    for slope_bound_idx, subgroup_means in enumerate(group_means):
        point_design = MPLT_COLOURS[(slope_bound_idx * 2) % len(MPLT_MARKERS)] + \
                       MPLT_MARKERS[slope_bound_idx % len(MPLT_MARKERS)]
        ax.plot(aspect_bounds_rad, subgroup_means, point_design,
                label=get_slope_label(slope_groups_bounds, slope_bound_idx))

    ax.legend(loc='lower right')
    ax.tick_params(axis='y', rotation=45)

    plt.savefig("original.png")
    plt.show()


def partition(arr, groups_count, upper_bound, lower_bound=0):
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
    slope_groups = partition(slope_bytes, slope_groups_count, upper_bound=slope_max_deg)
    aspect_groups = partition(aspect_bytes, aspect_groups_count, upper_bound=aspect_max_deg)
    groups = np.vstack((slope_groups, aspect_groups))

    group_means = npg.aggregate(groups, img_bytes, func='mean', fill_value=0)
    return group_means


def open_tiff(input_path):
    ds = gdal.Open(input_path, GA_ReadOnly)

    if ds is None:
        raise IOError(f"Wrong path: {input_path}")

    return ds.GetRasterBand(1).ReadAsArray().ravel()


img_bytes = open_tiff("..\\test_files\CORRECTED_3.tif")
slope_bytes = open_tiff("..\\test_files\SLOPE_3.tif")
aspect_bytes = open_tiff("..\\test_files\ASPECT_3.tif")

res = calculate_rose_groups(img_bytes, slope_bytes, aspect_bytes)
plot_rose_diagram(res)
