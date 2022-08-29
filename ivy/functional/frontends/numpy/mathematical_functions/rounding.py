import ivy
from ivy.func_wrapper import from_zero_dim_arrays_to_float


@from_zero_dim_arrays_to_float
def floor(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
    signature=None,
    extobj=None,
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    else:
        dtype = x.dtype
    if out is None:
        out = ivy.empty(x.shape)
    return ivy.astype(ivy.where(where, ivy.floor(x, out=x), out, out=out), dtype)


floor.unsupported_dtypes = {"torch": ("float16",)}
