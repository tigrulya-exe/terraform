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

from PyQt5.QtGui import QIcon
from qgis.core import QgsProcessingProvider

from .processing_alg.topocorrection.qgis_algorithm import TerraformTopoCorrectionAlgorithm
from .processing_alg.topocorrection_eval.correlation_eval import CorrelationEvaluationProcessingAlgorithm
from .processing_alg.topocorrection_eval.multi_criteria_eval import MultiCriteriaEvaluationProcessingAlgorithm
from .processing_alg.topocorrection_eval.multi_criteria_rank import MultiCriteriaRankProcessingAlgorithm
from .processing_alg.topocorrection_eval.plot_correlation_eval import PlotCorrelationEvaluationProcessingAlgorithm
from .processing_alg.topocorrection_eval.rose_diagram_eval import RoseDiagramEvaluationProcessingAlgorithm


class TerraformProcessingProvider(QgsProcessingProvider):

    def __init__(self):
        """
        Default constructor.
        """
        QgsProcessingProvider.__init__(self)

    def unload(self):
        """
        Unloads the provider. Any tear-down steps required by the provider
        should be implemented here.
        """
        pass

    def loadAlgorithms(self):
        """
        Loads all algorithms belonging to this provider.
        """
        self.addAlgorithm(TerraformTopoCorrectionAlgorithm())
        self.addAlgorithm(CorrelationEvaluationProcessingAlgorithm())
        self.addAlgorithm(PlotCorrelationEvaluationProcessingAlgorithm())
        self.addAlgorithm(RoseDiagramEvaluationProcessingAlgorithm())
        self.addAlgorithm(MultiCriteriaRankProcessingAlgorithm())
        self.addAlgorithm(MultiCriteriaEvaluationProcessingAlgorithm())

    def id(self):
        """
        Returns the unique provider id, used for identifying the provider. This
        string should be a unique, short, character only string, eg "qgis" or
        "gdal". This string should not be localised.
        """
        return 'Terraform'

    def name(self):
        """
        Returns the provider name, which is used to describe the provider
        within the GUI.

        This string should be short (e.g. "Lastools") and localised.
        """
        return self.tr('Terraform')

    def icon(self):
        """
        Should return a QIcon which is used for your provider inside
        the Processing toolbox.
        """
        return QIcon(":/plugins/terraform/icon.svg")

    def longName(self):
        """
        Returns the a longer version of the provider name, which can include
        extra details such as version numbers. E.g. "Lastools LIDAR tools
        (version 2.2.1)". This string should be localised. The default
        implementation returns the same string as name().
        """
        return self.name()
