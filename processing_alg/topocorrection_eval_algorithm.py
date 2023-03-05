from typing import Dict, Any

from qgis.PyQt.QtCore import QCoreApplication
from qgis._core import QgsProcessingParameterFileDestination, QgsProcessingParameterEnum, QgsProcessingParameterNumber
from qgis.core import (QgsProcessingContext,
                       QgsProcessingFeedback,
                       QgsProcessingParameterRasterLayer)

from processing_alg.execution_context import QgisExecutionContext
from processing_alg.terraform_algorithm import TerraformProcessingAlgorithm
from processing_alg.topocorrection_eval.eval_algorithm import TopoCorrectionEvalAlgorithm
from processing_alg.topocorrection_eval.regression_eval import RegressionEvalAlgorithm
from processing_alg.topocorrection_eval.rose_diagram_eval import RoseDiagramEvalAlgorithm


class TopocorrectionEvaluationAlgorithm(TerraformProcessingAlgorithm):
    def __init__(self):
        super().__init__()
        # todo dynamically scan directory
        self.algorithms = self.find_algorithms()

    def find_algorithms(self) -> dict[str, TopoCorrectionEvalAlgorithm]:
        algorithms = [
            RoseDiagramEvalAlgorithm,
            RegressionEvalAlgorithm
        ]
        algorithms_dict = dict()
        for algorithm in algorithms:
            algorithms_dict[algorithm.get_name()] = algorithm()
        return algorithms_dict

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        # Must return a new copy of your algorithm.
        return TopocorrectionEvaluationAlgorithm()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'terraform_topocorrection_eval'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Topocorrection evaluation algorithms')

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr('Contains several topocorrection algorithm evaluation methods.')

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
                self.tr('Input DEM layer')
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                'TOPO_CORRECTION_EVAL_ALGORITHM',
                self.tr('Topological correction evaluation algorithm'),
                options=self.algorithms.keys(),
                allowMultiple=False,
                defaultValue=RoseDiagramEvalAlgorithm.get_name(),
                usesStaticStrings=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SZA',
                self.tr('Solar zenith angle'),
                defaultValue=57.2478878065826,
                type=QgsProcessingParameterNumber.Double,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'SOLAR_AZIMUTH',
                self.tr('Solar azimuth'),
                defaultValue=177.744663052425,
                type=QgsProcessingParameterNumber.Double,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                'IMG_PLOT_OUT_PATH',
                self.tr('Output path of rose diagram'),
                fileFilter="*.png"
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
        eval_algorithm_name = self.parameterAsEnumString(parameters, 'TOPO_CORRECTION_EVAL_ALGORITHM', context)
        output_file_path = self.parameterAsFileOutput(parameters, 'IMG_PLOT_OUT_PATH', context)
        solar_zenith_angle = self.parameterAsDouble(parameters, 'SZA', context)
        solar_azimuth = self.parameterAsDouble(parameters, 'SOLAR_AZIMUTH', context)

        ctx = QgisExecutionContext(
            qgis_context=context,
            qgis_feedback=feedback,
            qgis_params=parameters,
            input_layer=input_layer,
            dem_layer=dem_layer,
            output_file_path=output_file_path,
            sza_degrees=solar_zenith_angle,
            solar_azimuth_degrees=solar_azimuth
        )
        self.algorithms[eval_algorithm_name].process(ctx)
        return {}
