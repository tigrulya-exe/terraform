from typing import List

import numpy as np
from osgeo import gdal
from osgeo.gdal import Band
from osgeo.gdalconst import GA_ReadOnly

gdal.UseExceptions()


def open_img(path, access=GA_ReadOnly):
    ds = gdal.Open(path, access)

    if ds is None:
        raise IOError(f"Wrong path: {path}")

    return ds


def read_band_as_array(path: str, band_idx: int = 1):
    ds = open_img(path)
    return ds.GetRasterBand(band_idx).ReadAsArray()


def read_band_flat(path: str, band_idx: int = 1):
    return read_band_as_array(path, band_idx=band_idx).ravel()


def raster_linear_regression(x_path: str, y_path: str, x_band: int = 1, y_band: int = 1):
    x_flat = read_band_as_array(x_path, x_band).ravel()
    y_flat = read_band_as_array(y_path, y_band).ravel()

    res = np.polynomial.polynomial.polyfit(x_flat, y_flat, 1)
    return res


def compute_band_means(input_path: str) -> List[float]:
    ds = open_img(input_path)

    if ds is None:
        raise IOError(f"Wrong path: {input_path}")

    band_means = []
    for iBand in range(0, ds.RasterCount):
        band = ds.GetRasterBand(iBand + 1)

        pixel_sum = 0
        for i in range(band.YSize - 1, -1, -1):
            scanline = read_hline(band, i)
            pixel_sum += np.sum(scanline)

        band_means.append(pixel_sum / (band.XSize * band.YSize))

    ds = None
    return band_means


def read_hline(band: Band, y_offset: int):
    return band.ReadAsArray(
        xoff=0,
        yoff=y_offset,
        win_xsize=band.XSize,
        win_ysize=1,
        buf_xsize=band.XSize,
        buf_ysize=1
    )


def get_raster_type(path: str):
    band = open_img(path).GetRasterBand(1)
    return gdal.GetDataTypeName(band.DataType)


def copy_band_descriptions(source_path, destination_path):
    source_ds = gdal.Open(source_path, gdal.GA_ReadOnly)
    destination_ds = gdal.Open(destination_path, gdal.GA_Update)

    if source_ds.RasterCount != destination_ds.RasterCount:
        raise ValueError(
            f"{source_path} and {destination_path} should have equal number of bands for metadata transfer"
        )

    for band_idx in range(1, source_ds.RasterCount + 1):
        band_description = source_ds.GetRasterBand(band_idx).GetDescription()
        destination_ds.GetRasterBand(band_idx).SetDescription(band_description)

    # de-reference the datasets, which triggers gdal to save
    source_ds = None
    destination_ds = None
