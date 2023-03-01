from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

sequence = range(0, 10)


def global_seq_test():
    for element in sequence:
        print(element)
        if element > 5:
            print("end")
            return


def global_seq_test2():
    for element in sequence:
        print(element)

# global_seq_test()
# global_seq_test2()

import matplotlib.pyplot as plt
import numpy as np
import math
import numpy_groupies as npg
import timeit
import time

# # t = np.arange(0.0, 2.0, 0.01)
# # s = 1 + np.sin(2 * np.pi * t)
#
# t = [0, 34, 234, 317]
# s = [12, 23, 16, 19]
#
# # plt.axes(projection='polar')
# # plt.polar([math.radians(deg) for deg in t], s, 'bo')
# ax = plt.subplot(111, polar=True)
# ax.plot([math.radians(deg) for deg in t], s, 'bo')
# ax.set_rlabel_position(0)
# # fig.savefig("test.png")
# plt.show()

def divide_to_groups(groups_count, start=0, end=90):
    group_size = (end - start) // groups_count
    return [start + i * group_size for i in range(groups_count)]


def find_group_start_idx(groups, val):
    for idx in range(len(groups) - 1, -1, -1):
        if groups[idx] <= val:
            return idx
    return -1

def build_group_selector(groups):
    return lambda val: find_group_start_idx(groups, val)

MPLT_MARKERS = "ov^<>1235sp*X"
MPLT_COLOURS = "bgrcmyk"

def calculate_rose_groups(
        img_bytes,
        slope_bytes,
        aspect_bytes,
        slope_groups_count=3,
        aspect_groups_count=36):
    slope_groups_bounds = divide_to_groups(slope_groups_count)
    aspect_groups_bounds = divide_to_groups(aspect_groups_count, 0, 360)

    print("start selector")
    start = time.time()

    # slope_groups = np.array(list(map(build_group_selector(slope_groups_bounds), slope_bytes)))
    # aspect_groups = np.array(list(map(build_group_selector(aspect_groups_bounds), aspect_bytes)))

    slope_group_size = 90 / slope_groups_count
    aspect_group_size = 360 / aspect_groups_count

    slope_groups = (slope_bytes // slope_group_size).astype(int, copy=False)
    aspect_groups = (aspect_bytes // aspect_group_size).astype(int, copy=False)

    stop = time.time()
    print('Time selector: ', stop - start)

    groups = np.vstack((slope_groups, aspect_groups))

    print("start")
    start = time.time()
    group_means = npg.aggregate(groups, img_bytes, func='mean', fill_value=0)
    stop = time.time()
    print('Time: ', stop - start)

    plot_rose_diagram(group_means, slope_groups_bounds, aspect_groups_bounds)
    return group_means


def get_slope_label(slope_groups_bounds, idx):
    higher_bound = "+" if len(slope_groups_bounds) == idx + 1 else f"-{slope_groups_bounds[idx + 1]}"
    return f"{slope_groups_bounds[idx]}{higher_bound}"

def plot_rose_diagram(group_means, slope_groups_bounds, aspect_groups_bounds):
    aspect_bounds_rad = [math.radians(deg) for deg in aspect_groups_bounds]

    ax = plt.subplot(111, polar=True)
    ax.set_rlabel_position(0)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))

    for slope_bound_idx, subgroup_means in enumerate(group_means):
        point_design = MPLT_COLOURS[(slope_bound_idx * 2) % len(MPLT_MARKERS)] + \
                       MPLT_MARKERS[slope_bound_idx % len(MPLT_MARKERS)]
        ax.plot(aspect_bounds_rad[:len(subgroup_means)], subgroup_means, point_design,
                label=get_slope_label(slope_groups_bounds, slope_bound_idx))

    ax.legend(loc='lower right')
    plt.savefig("original.png")
    plt.show()


def open_tiff(input_path):
    ds = gdal.Open(input_path, GA_ReadOnly)

    if ds is None:
        raise IOError(f"Wrong path: {input_path}")

    return ds.GetRasterBand(1).ReadAsArray().ravel()


img_bytes = open_tiff("test_files\CORRECTED_3.tif")
slope_bytes = open_tiff("test_files\SLOPE_3.tif")
aspect_bytes = open_tiff("test_files\ASPECT_3.tif")

# img_bytes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# slope_bytes = np.array([12, 45, 56, 13, 80, 89, 18, 5, 30, 10])
# aspect_bytes = np.array([210, 210, 78, 36, 140, 210, 210, 210, 210, 210])

res = calculate_rose_groups(img_bytes, slope_bytes, aspect_bytes, slope_groups_count=3, aspect_groups_count=36)
# print(res)

