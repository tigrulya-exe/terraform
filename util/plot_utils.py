import functools

from matplotlib import pyplot as plt, colors, scale


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


def norm_from_scale(scale_name):
    """
    Automatically generate a norm class from *scale_cls*.
    This differs from `.colors.make_norm_from_scale` in the following points:
    - This function is not a class decorator, but directly returns a norm class
      (as if decorating `.Normalize`).
    - The scale is automatically constructed with ``nonpositive="mask"``, if it
      supports such a parameter, to work around the difference in defaults
      between standard scales (which use "clip") and norms (which use "mask").
    Note that ``make_norm_from_scale`` caches the generated norm classes
    (not the instances) and reuses them for later calls.  For example,
    ``type(_auto_norm_from_scale("log")) == LogNorm``.
    """
    # Actually try to construct an instance, to verify whether
    # ``nonpositive="mask"`` is supported.
    try:
        scale_cls = scale._scale_mapping[scale_name]
    except KeyError:
        raise ValueError(
            "Invalid norm str name; the following values are "
            "supported: {}".format(", ".join(scale._scale_mapping))
        ) from None

    try:
        norm = colors.make_norm_from_scale(
            functools.partial(scale_cls, nonpositive="mask").func)(
            colors.Normalize)()
    except TypeError:
        norm = colors.make_norm_from_scale(scale_cls)(
            colors.Normalize)()

    return type(norm)()
