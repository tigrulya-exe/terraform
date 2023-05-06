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
