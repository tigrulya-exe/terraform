from typing import Any

from matplotlib import pyplot as plt

from processing_alg.execution_context import QgisExecutionContext


def draw_subplots(
        subplot_info,
        subplot_func,
        subplots_in_row=4,
        output_file_path=None,
        show=True,
        figsize=(15, 15),
        subplot_kw=None):
    row_count = len(subplot_info) // subplots_in_row
    if len(subplot_info) % subplots_in_row != 0:
        row_count += 1

    fig, axes = plt.subplots(row_count, subplots_in_row, figsize=figsize, subplot_kw=subplot_kw)
    for idx, ax in enumerate(axes.flat):
        if idx >= len(subplot_info):
            fig.delaxes(ax)
            continue

        subplot_func(ax, idx)

    plt.tight_layout()
    if output_file_path is not None:
        plt.savefig(output_file_path)
    if show:
        plt.show()

    return fig, axes


class TopoCorrectionEvalAlgorithm:
    def __init__(self):
        self.ctx = None

    def process(self, ctx: QgisExecutionContext) -> Any:
        self.ctx = ctx
        return self.process_internal()

    @staticmethod
    def get_name() -> str:
        pass

    def process_internal(self) -> Any:
        pass
