#!/usr/bin/env python
""" Terraform QGIS plugin.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = 'Tigran Manasyan'
__copyright__ = '(C) 2023 by Tigran Manasyan'
__license__ = "GPLv3"

import multiprocessing
import os
import platform
import sys
from pathlib import Path
from typing import Optional

from processing import Processing
from qgis.core import (QgsApplication, QgsProcessingContext, QgsProcessingUtils, QgsProject, QgsRasterLayer,
                       QgsProcessingParameterRasterDestination, QgsProcessingFeedback, QgsProcessingProvider)

_WIN_QGIS_PATH = None


def set_multiprocessing_metadata():
    if platform.system() == 'Windows':
        global _WIN_QGIS_PATH
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


def get_project_tmp_dir():
    tmp_param = QgsProcessingParameterRasterDestination(name="foo")
    return str(Path(tmp_param.generateTemporaryDestination()).parent.absolute())


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


class SilentFeedbackWrapper(QgsProcessingFeedback):
    def __init__(self, delegate: QgsProcessingFeedback):
        super().__init__()
        self.delegate = delegate

    def pushVersionInfo(self, provider: Optional['QgsProcessingProvider'] = ...):
        pass

    def pushConsoleInfo(self, info: str):
        pass

    def pushDebugInfo(self, info: str):
        pass

    def pushCommandInfo(self, info: str):
        pass

    def pushInfo(self, info: str):
        pass

    def pushWarning(self, warning: str):
        pass

    def reportError(self, error: str, fatalError: bool = True):
        self.delegate.reportError(error, fatalError)

    def setProgressText(self, text: str):
        pass

    def processedCountChanged(self, processedCount: int):
        self.delegate.processedCountChanged(processedCount)

    def progressChanged(self, progress: float):
        self.delegate.progressChanged(progress)

    def canceled(self):
        self.delegate.canceled()

    def cancel(self):
        self.delegate.cancel()

    def setProcessedCount(self, processedCount: int):
        self.delegate.setProcessedCount(processedCount)

    def processedCount(self) -> int:
        return self.delegate.processedCount()

    def progress(self) -> float:
        return self.delegate.progress()

    def setProgress(self, progress: float):
        return self.delegate.setProgress(progress)

    def isCanceled(self) -> bool:
        return self.delegate.isCanceled()
