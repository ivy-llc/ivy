# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_out,
    handle_numpy_dtype,
    handle_numpy_casting,
    from_zero_dim_arrays_to_scalar,
)


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _conj(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.conj(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
def angle(z, deg=False):
    angle = ivy.angle(z, deg=deg)
    if deg and len(z.shape) == 0:
        angle = ivy.astype(angle, ivy.float64)
    return angle


@to_ivy_arrays_and_back
def imag(val):
    return ivy.imag(val)


@to_ivy_arrays_and_back
def real(val):
    return ivy.real(val)
