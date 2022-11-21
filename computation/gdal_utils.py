import statistics
from typing import List

from osgeo import gdal
from osgeo.gdal import Band
from osgeo.gdalconst import GA_ReadOnly
import numpy as np
from numpy.polynomial.polynomial import Polynomial

class OutOfCoreRegressor:
    def __init__(self):
        self.weights = np.array([])
        self.x_sum = 0.0
        self.x_square_sum = 0.0
        self.y_sum = 0.0
        self.xy_sum = 0.0
        self.elements_num = 0

    def partial_train(self, x, y):
        np.set_printoptions(threshold=np.inf)

        x_flat = x.ravel()
        y_flat = y.ravel()
        self.x_sum += np.sum(x_flat)
        self.y_sum += np.sum(y_flat)
        self.x_square_sum += np.dot(x_flat, x_flat.T)
        self.xy_sum += np.dot(x_flat, y_flat.T)
        self.elements_num += len(x_flat)


    def train(self):
        a = np.array([
            [self.x_square_sum, self.x_sum],
            [self.x_sum, self.elements_num],
        ], dtype='float')
        b = np.array([self.xy_sum, self.y_sum], dtype='float')
        return np.linalg.solve(a, b)


def raster_linear_regression_full(x_path: str, y_path: str):
    x_ds = gdal.Open(x_path, GA_ReadOnly)
    y_ds = gdal.Open(y_path, GA_ReadOnly)

    x_band = x_ds.GetRasterBand(1)
    x_flat = x_band.ReadAsArray().ravel()

    y_band = y_ds.GetRasterBand(1)
    y_flat = y_band.ReadAsArray().ravel()

    res = np.polynomial.polynomial.polyfit(x_flat, y_flat, 1)

    band_weights = [res]
    print(res)

    return band_weights

def raster_linear_regression(x_path: str, y_path: str, qgis_feedback) -> List[List[float]]:
    x_ds = gdal.Open(x_path, GA_ReadOnly)
    y_ds = gdal.Open(y_path, GA_ReadOnly)

    band_weights = []
    # todo generalise it
    x_band = x_ds.GetRasterBand(1)
    for iBand in range(1, y_ds.RasterCount + 1):
        y_band = y_ds.GetRasterBand(iBand)

        regressor = OutOfCoreRegressor()
        for i in range(y_band.YSize):
            x_scanline = read_hline(x_band, i)
            y_scanline = read_hline(y_band, i)

            regressor.partial_train(x_scanline, y_scanline, qgis_feedback)

        weights = regressor.train()
        band_weights.append(weights)

    return band_weights


def compute_band_means(input_path: str) -> List[float]:
    ds = gdal.Open(input_path, GA_ReadOnly)

    if ds is None:
        raise IOError(f"Wrong path: {input_path}")

    band_means = []
    for iBand in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(iBand)

        line_means = []
        for i in range(band.YSize - 1, -1, -1):
            scanline = read_hline(band, i)
            line_means.append(np.mean(scanline))

        band_mean = statistics.mean(line_means)
        band_means.append(band_mean)

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

