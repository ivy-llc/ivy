import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back,handle_numpy_dtype

@handle_numpy_dtype
@to_ivy_arrays_and_back
def isreal(
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
    ret = ivy.isreal(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret