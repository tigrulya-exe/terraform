import os
from typing import Any, Dict

import numpy as np
from osgeo import gdal, gdal_array
from osgeo_utils.auxiliary.util import GetOutputDriverFor
from qgis._core import QgsProcessingParameterBoolean
from qgis.core import QgsProcessingParameterFolderDestination
from qgis.core import QgsProcessingParameterRasterLayer, \
    QgsProcessingFeedback, QgsProcessingParameterNumber

from .eval import EvaluationAlgorithm, MergeStrategy, PerFileMergeStrategy
from .qgis_algorithm import TopocorrectionEvaluationAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation import gdal_utils
from ...computation.qgis_utils import set_layers_to_load


class CorrelationNodeInfo:
    def __init__(
            self,
            histogram,
            name,
            x_bytes,
            img_stats,
            fit_stats=None):
        self.histogram = histogram
        self.name = name
        self.x_bytes = x_bytes
        self.img_stats = img_stats
        self.fit_stats = fit_stats


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
        band.WriteArray(histogram[::-1, ::-1])


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

        self.img_ds = gdal_utils.open_img(ctx.input_layer.source())
        self.luminance_bytes = gdal_utils.read_band_as_array(luminance_path).ravel()

    def evaluate_band(self, band_bytes, band_idx, gdal_band, group_idx) -> Any:
        x_min, x_max = 0, 1

        group_luminance_bytes = self.luminance_bytes[self.groups_map == group_idx]

        # todo change to band.minmax()
        img_min, img_max, *_ = gdal_band.GetStatistics(True, True)
        histogram, _, _ = np.histogram2d(
            group_luminance_bytes,
            band_bytes,
            bins=self.bins,
            range=[[x_min, x_max], [img_min, img_max]]
        )

        intercept, slope = np.polynomial.polynomial.polyfit(group_luminance_bytes, band_bytes, 1)
        return CorrelationNodeInfo(
            histogram.T,
            gdal_band.GetDescription(),
            group_luminance_bytes,
            (img_min, img_max),
            (intercept, slope)
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

    def need_to_show_result(self, execution_ctx: QgisExecutionContext):
        # In this algorithm we manually set output file(s)
        return False

    def add_output_param(self):
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                'OUTPUT_DIR',
                self.tr('Output directory'),
                createByDefault=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                'OPEN_OUT_FILE',
                self.tr('Open generated files in the output directory after running algorithm'),
                defaultValue=True
            )
        )

        return 'OUTPUT_DIR'

    def add_layers_to_project(self, ctx: QgisExecutionContext, results):
        need_open = self.parameterAsBoolean(ctx.qgis_params, 'OPEN_OUT_FILE', ctx.qgis_context)
        if need_open:
            set_layers_to_load(ctx.qgis_context, results)

    def processAlgorithmInternal(
            self,
            parameters: Dict[str, Any],
            context: QgisExecutionContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        context.sza_degrees = self.parameterAsDouble(parameters, 'SZA', context.qgis_context)
        context.solar_azimuth_degrees = self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context.qgis_context)

        luminance_path = context.calculate_luminance()
        if context.qgis_feedback.isCanceled():
            return {}

        group_ids_layer = self.parameterAsRasterLayer(parameters, 'CLASSIFICATION_MAP', context.qgis_context)
        group_ids_path = None if group_ids_layer is None else group_ids_layer.source()
        paths_with_names = self.compute_correlation(context, luminance_path, group_ids_path)

        self.add_layers_to_project(context, paths_with_names)
        return {}

    def get_output_dir(self, ctx: QgisExecutionContext):
        output_directory = self.parameterAsString(ctx.qgis_params, 'OUTPUT_DIR', ctx.qgis_context)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        return output_directory

    def compute_correlation(self, ctx: QgisExecutionContext, luminance_path, group_ids_path):
        bins = self.parameterAsInt(ctx.qgis_params, 'BIN_COUNT', ctx.qgis_context)
        output_directory = self.get_output_dir(ctx)

        def generate_file_name(node: CorrelationNodeInfo):
            return f"correlation_{node.name}.tif"

        merge_strategy = CorrelationPerFileMergeStrategy(
            output_directory,
            generate_file_name
        )

        alg = CorrelationEvaluationAlgorithm(
            ctx,
            merge_strategy,
            luminance_path,
            bins,
            group_ids_path,
        )

        output_paths = alg.evaluate()

        return [(path, os.path.basename(path)) for path in output_paths]
