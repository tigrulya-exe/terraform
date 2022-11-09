from math import cos, radians
from typing import Dict, Any

from osgeo import gdal
from qgis.PyQt.QtCore import QCoreApplication
from qgis._core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback, QgsCoordinateTransform, \
    QgsProcessingUtils, QgsProject
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingOutputNumber,
                       QgsProcessingParameterDistance,
                       QgsProcessingException,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterRasterDestination)
import processing


class TopoCorrectionContext:
    def __init__(
            self,
            qgis_context: QgsProcessingContext,
            qgis_feedback: QgsProcessingFeedback,
            qgis_params: Dict[str, Any],
            input_layer: QgsRasterLayer,
            slope_path: str,
            aspect_path: str,
            luminance_path: str,
            solar_zenith_angle: float,
            solar_azimuth: float):
        self.qgis_context = qgis_context
        self.qgis_params = qgis_params
        self.qgis_feedback = qgis_feedback
        self.input_layer = input_layer
        self.slope_path = slope_path
        self.aspect_path = aspect_path
        self.luminance_path = luminance_path
        self.solar_zenith_angle = solar_zenith_angle
        self.solar_azimuth = solar_azimuth

    def sza_cosine(self):
        cos(radians(self.solar_zenith_angle))

    def azimuth_cosine(self):
        cos(radians(self.solar_azimuth))


class ExampleProcessingAlgorithm(QgsProcessingAlgorithm):
    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        # Must return a new copy of your algorithm.
        return ExampleProcessingAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'topographic_correction'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Topographic Correction')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr('NSU')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs
        to.
        """
        return 'nsu'

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Topographically correct provided input layer.')

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and outputs of the algorithm.
        """
        # 'INPUT' is the recommended name for the main input
        # parameter.
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                'INPUT',
                self.tr('Input raster layer')
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                'DEM',
                self.tr('Input DEM layer'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Solar zenith angle')
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar azimuth')
            )
        )

        # 'OUTPUT' is the recommended name for the main output
        # parameter.
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                'OUTPUT',
                self.tr('Raster output')
            )
        )

    def processAlgorithm(
            self,
            parameters: Dict[str, Any],
            context: QgsProcessingContext,
            feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        input_layer = self.parameterAsRasterLayer(parameters, 'INPUT', context)
        dem_layer = self.parameterAsRasterLayer(parameters, 'DEM', context)
        solar_zenith_angle = self.parameterAsDouble(parameters, 'SZA', context)
        solar_azimuth = self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context)

        slope_path = self.build_slope_layer(feedback, context, dem_layer)
        if feedback.isCanceled():
            return {}

        aspect_path = self.build_aspect_layer(feedback, context, dem_layer)
        if feedback.isCanceled():
            return {}

        luminance_path = self.compute_luminance(
            feedback, context, slope_path, aspect_path, solar_zenith_angle, solar_azimuth)

        QgsProcessingUtils.mapLayerFromString(luminance_path, context)
        luminance_layer = QgsRasterLayer(luminance_path, "Luminance")
        luminance_layer.setExtent(input_layer.extent())
        QgsProject.instance().layerTreeRoot().addLayer(luminance_layer)

        if feedback.isCanceled():
            return {}

        topo_context = TopoCorrectionContext(
            context,
            feedback,
            parameters,
            input_layer,
            slope_path,
            aspect_path,
            luminance_path,
            solar_zenith_angle,
            solar_azimuth
        )

        return self.cosine_topocorrection(topo_context)

    def cosine_topocorrection(self, context: TopoCorrectionContext) -> Dict[str, Any]:
        result_bands = []
        luminance_layer = QgsRasterLayer(context.luminance_path, "Luminance layer")

        # use here gdal utils with ts=(1500, 1500)

        processing.run("gdal:warpreproject", {'INPUT': 'D:/Diploma/TestData/Kamchatka/ASPECT_CUT.tif',
                                              'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
                                              'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:32657'), 'RESAMPLING': 0,
                                              'NODATA': None, 'TARGET_RESOLUTION': 10, 'OPTIONS': '', 'DATA_TYPE': 0,
                                              'TARGET_EXTENT': None, 'TARGET_EXTENT_CRS': None, 'MULTITHREADING': False,
                                              'EXTRA': '', 'OUTPUT': 'TEMPORARY_OUTPUT'})

        for band_id in range(context.input_layer.bandCount()):
            try:
                results = processing.runAndLoadResults(
                    'gdal:rastercalculator',
                    {
                        # create layer from luminance_layer
                        'INPUT_A': luminance_layer,
                        'BAND_A': 1,
                        'INPUT_B': context.input_layer,
                        'BAND_B': band_id + 1,
                        'FORMULA': f'((B*{context.sza_cosine()})/cos(A))',
                        'OUTPUT': 'TEMPORARY_OUTPUT'
                    })
                result_bands.append(results['OUTPUT'])
            except QgsProcessingException as exc:
                raise RuntimeError(f"Error during performing topocorrection: {exc}")

            if context.qgis_feedback.isCanceled():
                return {}

        return processing.runAndLoadResults(
            "gdal:merge",
            {
                'INPUT': result_bands,
                'PCT': False,
                'SEPARATE': True,
                'NODATA_INPUT': None,
                'NODATA_OUTPUT': None,
                'OPTIONS': '',
                'EXTRA': '',
                'DATA_TYPE': 5,
                'OUTPUT': context.qgis_params['OUTPUT']
            },
            feedback=context.qgis_feedback,
            context=context.qgis_context
        )

    def build_slope_layer(self, feedback, context, dem_layer) -> str:
        results = processing.run(
            "gdal:slope",
            {
                'INPUT': dem_layer,
                'BAND': 1,
                # magic number lol
                'SCALE': 111120,
                'AS_PERCENT': False,
                'COMPUTE_EDGES': True,
                'ZEVENBERGEN': False,
                'OPTIONS': '',
                'EXTRA': '',
                'OUTPUT': 'TEMPORARY_OUTPUT'
            },
            feedback=feedback,
            context=context,
            is_child_algorithm=True
        )

        return results['OUTPUT']

    def build_aspect_layer(self, feedback, context, dem_layer) -> str:
        results = processing.run(
            "gdal:aspect",
            {
                'INPUT': dem_layer,
                'BAND': 1,
                'TRIG_ANGLE': True,
                'ZERO_FLAT': True,
                'COMPUTE_EDGES': True,
                'ZEVENBERGEN': False,
                'OPTIONS': '',
                'EXTRA': '',
                'OUTPUT': 'TEMPORARY_OUTPUT'
            },
            feedback=feedback,
            context=context,
            is_child_algorithm=True
        )

        return results['OUTPUT']

    def compute_luminance(self, feedback, context, slope_path: str, aspect_path: str, sza: float, solar_azimuth: float) -> str:
        sza_radians = radians(sza)
        solar_azimuth_radians = radians(solar_azimuth)

        results = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': slope_path,
                'BAND_A': 1,
                'INPUT_B': aspect_path,
                'BAND_B': 1,
                'FORMULA': f'(cos({sza_radians})*cos(deg2rad(A)) + '
                           f'sin({sza_radians})*sin(deg2rad(A))*cos(deg2rad(B) - {solar_azimuth_radians}))',
                'OUTPUT': 'TEMPORARY_OUTPUT'
            },
            feedback=feedback,
            context=context,
            is_child_algorithm=True
        )

        return results['OUTPUT']
