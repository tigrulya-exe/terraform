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
        return 'tigran_luminance'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Tigran luminance')

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
        return 'tigran_luminance'

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Example algorithm short description')

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
            QgsProcessingParameterRasterLayer(
                'SLOPE',
                self.tr('Input slope layer'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                'ASPECT',
                self.tr('Input aspect layer'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Input aspect layer')
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar Azimuth')
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

        return self.compute_luminance(parameters, context, slope_layer, aspect_layer)

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
        sza = self.parameterAsDouble(parameters,
                                     'SZA',
                                     context)

        solar_azimuth = self.parameterAsDouble(parameters,
                                               'SOLAR_AZIMUTH',
                                               context)

        print(f'(cos({sza})*cos(A) + sin({sza})*sin(A)*cos(B - {solar_azimuth}))')
        results = processing.runAndLoadResults(
            'gdal:rastercalculator',
            {
                'INPUT_A': slope_layer,
                'BAND_A': 1,
                'INPUT_B': aspect_layer,
                'BAND_B': 1,
                'FORMULA': f'(cos({sza})*cos(A) + sin({sza})*sin(A)*cos(B - {solar_azimuth}))',
                'OUTPUT': parameters['OUTPUT']
            })

        return {'OUTPUT': results['OUTPUT']}
