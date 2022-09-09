# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


@from_zero_dim_arrays_to_float
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


def fix(
    x,
    /,
    out=None,
):
    where = ivy.greater_equal(x, 0)
    return ivy.where(where, ivy.floor(x, out=out), ivy.ceil(x, out=out), out=out)
