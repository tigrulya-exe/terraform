from math import cos, radians

from qgis.PyQt.QtCore import QCoreApplication
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


class ExampleProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer,
    creates some new layers and returns some results.
    """

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
        return 'luminance_map'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Luminance map')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr('By Tigran')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs
        to.
        """
        return 'tigran'

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Builds luminance map')

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

    def processAlgorithm(self, parameters, context, feedback):
        input_layer = self.parameterAsRasterLayer(parameters,
                                                  'INPUT',
                                                  context)

        dem_layer = self.parameterAsRasterLayer(parameters,
                                                'DEM',
                                                context)

        slope_layer = self.build_slope_layer(dem_layer)

        if feedback.isCanceled():
            return {}

        aspect_layer = self.build_aspect_layer(dem_layer)

        if feedback.isCanceled():
            return {}

        sza = self.parameterAsDouble(parameters,
                                     'SZA',
                                     context)

        sza_radians = radians(sza)

        luminance = self.compute_luminance(parameters, context, slope_layer, aspect_layer)

        return self.cosine_topocorrection(parameters, luminance, input_layer, sza_radians)

    def cosine_topocorrection(self, parameters, luminance_layer, input_layer, sza_radians):
        sza_cosine = cos(sza_radians)

        result_bands = []
        for band_id in range(input_layer.bandCount()):
            results = processing.runAndLoadResults(
                'gdal:rastercalculator',
                {
                    # create layer from luminance_layer
                    'INPUT_A': luminance_layer,
                    'BAND_A': 1,
                    'INPUT_B': input_layer,
                    'BAND_B': band_id + 1,
                    'FORMULA': f'((B*{sza_cosine})/cos(A))',
                    'OUTPUT': 'TEMPORARY_OUTPUT'
                })
            # тут как я понял мы можем убрать dataProvider().dataSourceUri(), т.к.
            # в аутпуте пхду лежит строка
            result_bands.append(results['OUTPUT'].dataProvider().dataSourceUri())

        #     todo merge bands into layer
        return processing.run(
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
                'OUTPUT':  parameters['OUTPUT']
            })

    def build_slope_layer(self, dem_layer):
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
            })

        return results['OUTPUT']

    def build_aspect_layer(self, dem_layer):
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
            })

        return results['OUTPUT']

    def compute_luminance(self, parameters, context, slope_layer, aspect_layer):
        # todo
        sza = self.parameterAsDouble(parameters,
                                     'SZA',
                                     context)
        sza_radians = radians(sza)

        solar_azimuth = self.parameterAsDouble(parameters,
                                               'SOLAR_AZIMUTH',
                                               context)
        solar_azimuth_radians = radians(solar_azimuth)

        results = processing.runAndLoadResults(
            'gdal:rastercalculator',
            {
                'INPUT_A': slope_layer,
                'BAND_A': 1,
                'INPUT_B': aspect_layer,
                'BAND_B': 1,
                'FORMULA': f'(cos({sza_radians})*cos(deg2rad(A)) + '
                           f'sin({sza_radians})*sin(deg2rad(A))*cos(deg2rad(B) - {solar_azimuth_radians}))',
                'OUTPUT': parameters['OUTPUT']
            })

        return {'OUTPUT': results['OUTPUT']}
