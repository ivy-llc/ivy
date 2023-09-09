import math


class FiniteStatus:
    all_finite = 0
    has_nan = 1
    has_infinite = 2


# --- Helpers --- #
# --------------- #


def _isfinite_allow_nan(a):
    for v in a:
        if math.isinf(v):
            return FiniteStatus.has_infinite
    return FiniteStatus.all_finite


def _isfinite_disable_nan(a):
    for v in a:
        if math.isnan(v):
            return FiniteStatus.has_nan
        elif math.isinf(v):
            return FiniteStatus.has_infinite
    return FiniteStatus.all_finite


# --- Main --- #
# ------------ #


def cy_isfinite(a, allow_nan=False):
    result = FiniteStatus.all_finite
    if allow_nan:
        result = _isfinite_allow_nan(a)
    else:
        result = _isfinite_disable_nan(a)
    return result
