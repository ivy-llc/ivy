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
def minkowski(u: List[float], /, *, v: List[float], p: float = 2, w: Optional[float] = None) -> float:
    u,v = _validate_vector(u),_validate_vector(v)
    if p <= 0:
        raise ValueError("p must be greater than 0")
    u_v = u - v
    root_w = {1: w, 2: ivy.sqrt(w), ivy.inf: w != 0}.get(p, w ** (1/p) if w else None)
    u_v = [ivy.multiply(root_w, elem) if root_w is not None else elem for elem in u_v]
    return ivy_np.linalg.norm(u_v, ord=p)


# euclidean
@to_ivy_arrays_and_back
def euclidean(u, v, /, *, w=None):
    return minkowski(u, v, p=2, w=w)
