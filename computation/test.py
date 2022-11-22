from typing import List

from numpy.polynomial import Polynomial
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import numpy as np

from computation.gdal_utils import read_hline


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
        if np.isinf(self.x_sum):
            print(f'gg{self.elements_num}-{x_flat} ++++++ {y_flat}')

        self.y_sum += np.sum(y_flat)
        self.x_square_sum += np.dot(x_flat, x_flat.T)
        self.xy_sum += np.dot(x_flat, y_flat.T)
        self.elements_num += len(x_flat)
        # print(f'resssss -> {self.x_sum}-{self.y_sum}-{self.x_square_sum}-{self.xy_sum}-{self.elements_num}')

    def train(self):
        a = np.array([
            [self.x_square_sum, self.x_sum],
            [self.x_sum, self.elements_num],
        ], dtype='float')
        b = np.array([self.xy_sum, self.y_sum], dtype='float')
        return np.linalg.solve(a, b)



def raster_linear_regression(x_path: str, y_path: str) -> List[List[float]]:
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

            regressor.partial_train(x_scanline, y_scanline)

        weights = regressor.train()
        band_weights.append(weights)
        print(weights)

    return band_weights

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


def create_test_file(outFileName, xsize=200, ysize=100, generator=lambda x, y: x*y):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outFileName, xsize=xsize, ysize=ysize, bands=1, eType=gdal.GDT_Int32)

    raster = np.array([[generator(x, y) for x in range(xsize)] for y in range(ysize)])

    outdata.GetRasterBand(1).WriteArray(raster)


def test():
    x_path = "./x.tif"
    y_path = "./y.tif"
    create_test_file(x_path)
    create_test_file(y_path, generator=lambda x, y: 4 * y * x + 1)
    raster_linear_regression_full(x_path, y_path)

test()