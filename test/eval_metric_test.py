import unittest

import numpy as np
from osgeo import gdal

from processing_alg.topocorrection_eval.metrics import EvalMetric


# Todo update test
class TestEvalMetrics(unittest.TestCase):

    def test_norm_metrics(self):
        metric = EvalMetric(is_reduction=False)

        metric_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_normed_metric_vals = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # normed_metric_vals = metric.norm(metric_vals)
        normed_metric_vals = []
        np.testing.assert_almost_equal(normed_metric_vals, expected_normed_metric_vals)

    def test_reduction_norm_metrics(self):
        metric = EvalMetric(is_reduction=True)

        metric_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_normed_metric_vals = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        # normed_metric_vals = metric.norm(metric_vals)
        normed_metric_vals = []

        np.testing.assert_almost_equal(normed_metric_vals, expected_normed_metric_vals)


if __name__ == '__main__':
    unittest.main()

# TODO
def create_test_file(outFileName, xsize=200, ysize=100, generator=lambda x, y: x * y):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outFileName, xsize=xsize, ysize=ysize, bands=1, eType=gdal.GDT_Int32)

    raster = np.array([[generator(x, y) for x in range(xsize)] for y in range(ysize)])

    outdata.GetRasterBand(1).WriteArray(raster)
