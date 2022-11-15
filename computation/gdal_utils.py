import statistics
from typing import List

from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import numpy as np

def compute_band_means(input_path: str) -> List[float]:
    ds = gdal.Open(input_path, GA_ReadOnly)

    if ds is None:
        raise IOError(f"Wrong path: {input_path}")

    band_means = []
    for iBand in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(iBand)

        line_means = []
        for i in range(band.YSize - 1, -1, -1):
            scanline = band.ReadAsArray(
                xoff=0,
                yoff=i,
                win_xsize=band.XSize,
                win_ysize=1,
                buf_xsize=band.XSize,
                buf_ysize=1
            )
            line_means.append(np.mean(scanline))

        band_mean = statistics.mean(line_means)
        band_means.append(band_mean)

    return band_means
