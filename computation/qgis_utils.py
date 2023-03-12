from qgis.core import QgsProcessingContext, QgsProcessingUtils, QgsProject, QgsRasterLayer


def add_layer_to_project(context, layer_path, name="out"):
    context.addLayerToLoadOnCompletion(
        layer_path,
        QgsProcessingContext.LayerDetails(
            name,
            QgsProject.instance(),
            layerTypeHint=QgsProcessingUtils.LayerHint.Raster
        )
    )


def check_compatible(left_layer: QgsRasterLayer, right_layer: QgsRasterLayer):
    if left_layer.extent() != right_layer.extent():
        raise Exception(
            f"Extents of {left_layer.name()} and {right_layer.name()} should be the same"
        )

    if left_layer.crs() != right_layer.crs():
        raise Exception(
            f"Coordinate reference systems of {left_layer.name()} and {right_layer.name()} should be the same"
        )

    if left_layer.width() != right_layer.width() or left_layer.height() != right_layer.height():
        raise Exception(
            f"Resolutions of {left_layer.name()} and {right_layer.name()} should be the same"
        )
