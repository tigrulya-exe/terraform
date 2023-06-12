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
from typing import Dict, Any

from qgis.core import QgsProcessingParameterBoolean, QgsProcessingContext, QgsProcessingParameterNumber


class ParallelProcessingParamMixin:

    def get_run_parallel_param(self, parameters: Dict[str, Any], context: QgsProcessingContext):
        return self.parameterAsBoolean(parameters, 'RUN_PARALLEL', context)

    def get_parallel_timeout_param(self, parameters: Dict[str, Any], context: QgsProcessingContext):
        return self.parameterAsInt(parameters, 'TASK_TIMEOUT', context)

    def get_worker_count_param(self, parameters: Dict[str, Any], context: QgsProcessingContext):
        return self.parameterAsInt(parameters, 'WORKER_COUNT', context)

    def parallel_run_params(self):
        parallel_param = QgsProcessingParameterBoolean(
            'RUN_PARALLEL',
            'Run processing in parallel',
            defaultValue=False,
            optional=True
        )

        task_timeout_param = QgsProcessingParameterNumber(
            'TASK_TIMEOUT',
            'Parallel task timeout in ms',
            defaultValue=10000,
            type=QgsProcessingParameterNumber.Integer
        )

        worker_count = QgsProcessingParameterNumber(
            'WORKER_COUNT',
            'Number of workers for parallel processing',
            defaultValue=multiprocessing.cpu_count(),
            type=QgsProcessingParameterNumber.Integer
        )

        return parallel_param, task_timeout_param, worker_count
