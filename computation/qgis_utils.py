from qgis.core import QgsProcessingContext, QgsProcessingUtils, QgsProject


def add_layer_to_project(context, layer_path, name="out"):
    context.addLayerToLoadOnCompletion(
        layer_path,
        QgsProcessingContext.LayerDetails(
            name,
            QgsProject.instance(),
            layerTypeHint=QgsProcessingUtils.LayerHint.Raster
        )
    )
