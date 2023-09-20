import ivy
from ivy.func_wrapper import to_ivy_arrays_and_back


class FiniteStatus:
    all_finite = 0
    has_nan = 1
    has_infinite = 2


# --- Helpers --- #
# --------------- #


@to_ivy_arrays_and_back
def _isfinite_allow_nan(a):
    for v in a:
        if ivy.isinf(v):
            return FiniteStatus.has_infinite
    return FiniteStatus.all_finite


@to_ivy_arrays_and_back
def _isfinite_disable_nan(a):
    for v in a:
        if ivy.isnan(v):
            return FiniteStatus.has_nan
        elif ivy.isinf(v):
            return FiniteStatus.has_infinite
    return FiniteStatus.all_finite


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
def cy_isfinite(a, allow_nan=False):
    result = FiniteStatus.all_finite
    if allow_nan:
        result = _isfinite_allow_nan(a)
    else:
        result = _isfinite_disable_nan(a)
    return result
