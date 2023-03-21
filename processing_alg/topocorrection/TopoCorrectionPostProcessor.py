from qgis.core import QgsProcessingLayerPostProcessorInterface, QgsRasterLayer

from ...computation.gdal_utils import copy_band_descriptions


class TopoCorrectionPostProcessor(QgsProcessingLayerPostProcessorInterface):
    instance = None

    def __init__(self, input_layer) -> None:
        super().__init__()
        self.input_layer: QgsRasterLayer = input_layer

    def postProcessLayer(self, layer, context, feedback):
        layer.setRenderer(self.input_layer.renderer())
        copy_band_descriptions(self.input_layer.source(), layer.source())
        layer.reload()
        layer.triggerRepaint()

    # sip hack
    @staticmethod
    def create(input_layer) -> 'TopoCorrectionPostProcessor':
        TopoCorrectionPostProcessor.instance = TopoCorrectionPostProcessor(input_layer)
        return TopoCorrectionPostProcessor.instance
