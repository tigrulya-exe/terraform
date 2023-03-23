from enum import Enum

from qgis.core import QgsProcessingParameterEnum

from ..execution_context import QgisExecutionContext


class OutputFormatMixin:
    class OutputFormat(str, Enum):
        PNG = 'png'
        SVG = 'svg'

    def is_supported_by_qgis(self, fmt: OutputFormat):
        return fmt == self.OutputFormat.PNG

    def format_param_supported_by_qgis(self, ctx: QgisExecutionContext):
        fmt = self.get_output_format_param(ctx)
        return fmt == self.OutputFormat.PNG

    def get_output_format_param(self, ctx: QgisExecutionContext):
        return self.parameterAsEnumString(ctx.qgis_params, 'OUTPUT_FORMAT', ctx.qgis_context)

    def output_format_param(self):
        return QgsProcessingParameterEnum(
            'OUTPUT_FORMAT',
            'Output format',
            options=[f for f in self.OutputFormat],
            allowMultiple=False,
            defaultValue='png',
            usesStaticStrings=True
        )
