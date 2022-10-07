# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float
from ivy.functional.frontends.numpy import to_ivy_arrays_and_back


@from_zero_dim_arrays_to_float
@to_ivy_arrays_and_back
def ceil(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.ceil(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@to_ivy_arrays_and_back
def fix(
    x,
    /,
    out=None,
):
    where = ivy.greater_equal(x, 0)
    return ivy.where(where, ivy.floor(x, out=out), ivy.ceil(x, out=out), out=out)
