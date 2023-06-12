from typing import List

import numpy as np
from osgeo import gdal
from osgeo.gdal import Band
from osgeo.gdalconst import GA_ReadOnly

gdal.UseExceptions()

GDAL_DATATYPES = [
    'Byte',
    'Int16',
    'UInt16',
    'UInt32',
    'Int32',
    'Float32',
    'Float64',
    'CInt16',
    'CInt32',
    'CFloat32',
    'CFloat64'
]


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
        _, _, mean, _ = band.GetStatistics(True, True)

        if mean is not None:
            band_means.append(mean)
            continue

        mean = np.mean(band.ReadAsArray().astype('float'))
        band_means.append(mean)

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


def get_raster_type_ordinal(raster_type: str):
    return GDAL_DATATYPES.index(raster_type)
