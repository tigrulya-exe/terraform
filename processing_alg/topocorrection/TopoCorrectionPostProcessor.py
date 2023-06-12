from osgeo import gdal
from qgis.core import QgsProcessingLayerPostProcessorInterface, QgsRasterLayer


class TopoCorrectionPostProcessor(QgsProcessingLayerPostProcessorInterface):
    instance = None

    def __init__(self, input_layer) -> None:
        super().__init__()
        self.input_layer: QgsRasterLayer = input_layer

    # sip hack
    @staticmethod
    def create(input_layer) -> 'TopoCorrectionPostProcessor':
        TopoCorrectionPostProcessor.instance = TopoCorrectionPostProcessor(input_layer)
        return TopoCorrectionPostProcessor.instance

    def postProcessLayer(self, layer, context, feedback):
        layer.setRenderer(self.input_layer.renderer().clone())
        self._copy_band_descriptions(self.input_layer.source(), layer.source())
        layer.reload()
        layer.triggerRepaint()

    def _copy_band_descriptions(self, source_path, destination_path):
        source_ds = gdal.Open(source_path, gdal.GA_ReadOnly)
        destination_ds = gdal.Open(destination_path, gdal.GA_Update)

        if source_ds.RasterCount != destination_ds.RasterCount:
            raise ValueError(
                f"{source_path} and {destination_path} should have equal number of bands for metadata transfer"
            )

        for band_idx in range(1, source_ds.RasterCount + 1):
            band_description = source_ds.GetRasterBand(band_idx).GetDescription()
            destination_ds.GetRasterBand(band_idx).SetDescription(band_description)
