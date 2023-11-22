# global
import ivy

# local
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_out,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_casting,
)


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _nextafter(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    return ivy.nextafter(x1, x2, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _signbit(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="safe",
    order="K",
    dtype=None,
    subok=True,
):
    x = ivy.astype(x, ivy.float64)
    return ivy.logical_or(ivy.less(x, 0), ivy.atan2(0.0, x) == ivy.pi, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _spacing(
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
    # Implement the frontend function using Ivy compositions
    if dtype is None:
        dtype = ivy.dtype(x)
    y = ivy.floor(ivy.log2(ivy.abs(x + 1)))
    spacing = ivy.multiply(ivy.finfo(dtype).eps, ivy.pow(2, y))
    if dtype != "float16":
        spacing = ivy.sign(x) * spacing
    return spacing
