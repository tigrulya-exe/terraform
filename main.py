from qgis.core import *
import processing
from processing.core.Processing import Processing
import inspect

Processing.initialize()

project = QgsProject.instance()
print(project.fileName())
project.read(r"D:\Diploma\QGisProjects\Project2.qgz")
original_layer = project.mapLayersByName("Kamchatka")[0]
print(original_layer.width(), original_layer.height())

dem = project.mapLayersByName("DEM_CUT")[0]
print(dem.width(), dem.height())

output_raster = r"D:/Diploma/TestData/Kamchatka/new.tif"
calc_parameters = {
    'INPUT_A': original_layer,
    'BAND_A': 1,
    'FORMULA': '(cos(A))',
    'OUTPUT': QgsProcessingOutputLayerDefinition(output_raster)
}

# 111120
alg = QgsApplication.processingRegistry().createAlgorithmById('gdal:rastercalculator')
# lines = inspect.getsource(alg.checkParameterValues)
# print(lines)
processing.runAndLoadResults('gdal:rastercalculator', calc_parameters)
