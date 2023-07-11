# global
import ivy
from ivy.functional.frontends.scipy.func_wrapper import (
    to_ivy_arrays_and_back,
)
import ivy.functional.frontends.scipy as sc_frontend


# Helpers #
# ------- #


def _validate_vector(u, dtype=None):
    u = ivy.asarray(u, dtype=dtype)
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")


def _validate_weights(w, dtype="float64"):
    w = _validate_vector(w, dtype=dtype)
    if ivy.any(w < 0):
        raise ValueError("Input weights should be all non-negative")
    return w


# Functions #
# --------- #


# minkowski
@to_ivy_arrays_and_back
def minkowski(u, v, p=2, /, *, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    if p <= 0:
        raise ValueError("p must be greater than 0")
    u_v = u - v
    if w is not None:
        w = _validate_weights(w)
        if p == 1:
            root_w = w
        elif p == 2:
            # better precision and speed
            root_w = ivy.sqrt(w)
        elif p == ivy.inf:
            root_w = w != 0
        else:
            root_w = ivy.pow(w, 1 / p)
        u_v = ivy.multiply(root_w, u_v)
    dist = sc_frontend.linalg.norm(u_v, ord=p)
    return dist


# euclidean
@to_ivy_arrays_and_back
def euclidean(u, v, /, *, w=None):
    return minkowski(u, v, p=2, w=w)
