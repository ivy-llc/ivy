# local
from numpy.lib.function_base import _quantile_unchecked, _check_interpolation_as_method

import ivy
from ivy.functional.frontends.numpy.statistics.averages_and_variances import (
    _quantile_is_valid,
)


def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None
):
    if interpolation is not None:
        method = _check_interpolation_as_method(method, interpolation, "quantile")

    q = ivy.array(q)

    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    return _quantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims)
