import multiprocessing
import os
import platform
import sys

from processing import Processing
from qgis.core import QgsApplication, QgsProcessingContext, QgsProcessingUtils, QgsProject, QgsRasterLayer

_WIN_QGIS_PATH = None


def set_multiprocessing_metadata():
    if platform.system() == 'Windows':
        global _WIN_QGIS_PATH
        # _WIN_QGIS_PATH = str(Path(sys.executable).parent.parent)
        _WIN_QGIS_PATH = os.environ["QGIS_PREFIX_PATH"]
        multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


def qgis_path():
    return _WIN_QGIS_PATH


def init_qgis_env(qgis_install_path):
    if platform.system() == 'Windows':
        # Initialize QGIS Application
        app = QgsApplication([], False)
        QgsApplication.setPrefixPath(qgis_install_path, True)
        QgsApplication.initQgis()
        Processing.initialize()
        return app


def add_layer_to_load(context, layer_path, name="out"):
    context.addLayerToLoadOnCompletion(
        layer_path,
        QgsProcessingContext.LayerDetails(
            name,
            QgsProject.instance(),
            layerTypeHint=QgsProcessingUtils.LayerHint.Raster
        )
    )


def set_layers_to_load(context, layers_with_names):
    layers_dict = {layer_path: QgsProcessingContext.LayerDetails(
        layer_name,
        QgsProject.instance(),
        layerTypeHint=QgsProcessingUtils.LayerHint.Raster
    ) for layer_path, layer_name in layers_with_names}

    context.setLayersToLoadOnCompletion(layers_dict)


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


def table_from_matrix_list(table_list, val_per_row=1):
    it = iter(table_list)
    return {key: [next(it) for _ in range(val_per_row)] for key in it}


def matrix_list_from_table(table_dict):
    result = []

    for key, values in table_dict.items():
        result.append(key)
        result.extend(values)

    return result
