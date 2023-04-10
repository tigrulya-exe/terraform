import os
from typing import Any, Dict

import numpy as np
from osgeo import gdal, gdal_array
from osgeo_utils.auxiliary.util import GetOutputDriverFor
from qgis.core import QgsProcessingParameterRasterLayer, \
    QgsProcessingFeedback, QgsProcessingParameterNumber, QgsProcessingContext

from .eval import EvaluationAlgorithm, MergeStrategy, PerFileMergeStrategy
from .qgis_algorithm import TopocorrectionEvaluationAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation import gdal_utils


class CorrelationNodeInfo:
    def __init__(
            self,
            histogram,
            name,
            x_bytes,
            img_stats,
            fit_stats=None,
            group_idx=None):
        self.histogram = histogram
        self.name = name
        self.x_bytes = x_bytes
        self.img_stats = img_stats
        self.fit_stats = fit_stats
        self.group_idx = group_idx


class CorrelationPerFileMergeStrategy(PerFileMergeStrategy):
    def __init__(self, output_directory: str, path_provider) -> None:
        super().__init__(output_directory, path_provider)

    def save_result(self, result: CorrelationNodeInfo, filename):
        histogram = result.histogram

        driver_name = GetOutputDriverFor(filename)
        driver = gdal.GetDriverByName(driver_name)

        out_ds = driver.Create(
            os.path.join(self.output_directory, filename),
            xsize=histogram.shape[0],
            ysize=histogram.shape[1],
            eType=gdal_array.NumericTypeCodeToGDALTypeCode(histogram.dtype))

        band = out_ds.GetRasterBand(1)
        # rotate matrix by 180 deg
        band.WriteArray(histogram[::-1, :])


class CorrelationEvaluationAlgorithm(EvaluationAlgorithm):
    def __init__(
            self,
            ctx: QgisExecutionContext,
            merge_strategy: MergeStrategy,
            luminance_path,
            bins=100,
            group_ids=None):
        super().__init__(ctx, merge_strategy, group_ids)
        self.bins = bins

        self.img_ds = gdal_utils.open_img(ctx.input_layer_path)
        self.luminance_bytes = gdal_utils.read_band_as_array(luminance_path).ravel()

    def _evaluate_band(self, band: EvaluationAlgorithm.BandInfo, group_idx) -> Any:
        x_min, x_max = 0, 1

        group_luminance_bytes = self.luminance_bytes[self.groups_map == group_idx]

        # todo change to band.minmax()
        img_min, img_max, *_ = band.gdal_band.GetStatistics(True, True)
        histogram, _, _ = np.histogram2d(
            group_luminance_bytes,
            band.bytes,
            bins=self.bins,
            range=[[x_min, x_max], [img_min, img_max]]
        )

        intercept, slope = np.polynomial.polynomial.polyfit(group_luminance_bytes, band.bytes, 1)
        return CorrelationNodeInfo(
            histogram.T,
            band.gdal_band.GetDescription(),
            group_luminance_bytes,
            (img_min, img_max),
            (intercept, slope),
            group_idx
        )


class CorrelationEvaluationProcessingAlgorithm(TopocorrectionEvaluationAlgorithm):
    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)

        self.addParameter(
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Solar zenith angle (in degrees)'),
                defaultValue=57.2478878065826,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar azimuth (in degrees)'),
                defaultValue=177.744663052425,
                type=QgsProcessingParameterNumber.Double
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'BIN_COUNT',
                self.tr('Number of bins per axis in 2d diagram'),
                defaultValue=100,
                type=QgsProcessingParameterNumber.Integer,
            )
        )

        classification_map_param = QgsProcessingParameterRasterLayer(
            'CLASSIFICATION_MAP',
            self.tr('Raster layer with classification label ids for input raster'),
            optional=True
        )
        self._additional_param(classification_map_param)

    def createInstance(self):
        return CorrelationEvaluationProcessingAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'luminance_radiance_correlation_eval'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Correlation between luminance and radiance')

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Builds density plots (2d histograms) of the relationship between '
                       'raster image bands and illumination model of that image.\n'
                       "<b>Note:</b> the illumination model of the input raster image is calculated automatically, "
                       "based on the provided DEM layer. Currently, the input raster image and the DEM must have "
                       "the same CRS, extent and spatial resolution.")

    def _process_internal(
            self,
            parameters: Dict[str, Any],
            context: QgisExecutionContext,
            feedback: QgsProcessingFeedback
    ):
        group_ids_layer = self.parameterAsRasterLayer(parameters, 'CLASSIFICATION_MAP', context.qgis_context)
        group_ids_path = None if group_ids_layer is None else group_ids_layer.source()
        paths_with_names = self._compute_correlation(context, group_ids_path)

        return paths_with_names

    def _ctx_additional_kw_args(
            self,
            parameters: Dict[str, Any],
            context: QgsProcessingContext) -> Dict[str, Any]:
        return {
            'sza_degrees': self.parameterAsDouble(parameters, 'SZA', context),
            'solar_azimuth_degrees': self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context),
        }

    def _compute_correlation(self, ctx: QgisExecutionContext, group_ids_path):
        bins = self.parameterAsInt(ctx.qgis_params, 'BIN_COUNT', ctx.qgis_context)

        def generate_file_name(node: CorrelationNodeInfo):
            return f"correlation_{node.group_idx}_{node.name}.tif"

        merge_strategy = CorrelationPerFileMergeStrategy(
            ctx.output_file_path,
            generate_file_name
        )

        alg = CorrelationEvaluationAlgorithm(
            ctx,
            merge_strategy,
            ctx.luminance_path,
            bins,
            group_ids_path,
        )

        output_paths = alg.evaluate()

        return [(path, os.path.basename(path)) for path in output_paths]
