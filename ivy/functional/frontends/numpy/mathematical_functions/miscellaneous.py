# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


def convolve(a, v, mode='full'):
    pass


@from_zero_dim_arrays_to_float
def clip(a, 
         a_min, 
         a_max, 
         /,
         out=None,
         *,
         where=True,
         casting="same_kind",
         order="k",
         dtype=None,
         subok=True,):
    
    if not dtype:
        dtype = a.dtype

    ret = ivy.minimum(a_max, ivy.maximum(a, a_min), out=out)
        
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    ret = ivy.astype(ivy.array(ret), ivy.as_ivy_dtype(dtype), out=out)

    return ret


# sqrt
@from_zero_dim_arrays_to_float
def sqrt(
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
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.sqrt(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@from_zero_dim_arrays_to_float
def cbrt(
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
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.pow(x, 1.0 / 3.0, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


def square(
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
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.square(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret
