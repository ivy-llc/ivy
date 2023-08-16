# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)
import ivy.functional.frontends.numpy as np_frontend


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def concatenate(arrays, /, *, axis=0, out=None, dtype=None, casting="same_kind"):
    if dtype is not None:
        out_dtype = ivy.as_ivy_dtype(dtype)
    else:
        out_dtype = ivy.dtype(arrays[0])
        for i in arrays:
            out_dtype = ivy.as_ivy_dtype(
                np_frontend.promote_numpy_dtypes(i.dtype, out_dtype)
            )
    return ivy.concat(arrays, axis=axis, out=out).astype(out_dtype, copy=False)


@handle_numpy_out
@to_ivy_arrays_and_back
def stack(arrays, /, *, axis=0, out=None):
    out_dtype = ivy.dtype(arrays[0])
    for i in arrays:
        out_dtype = ivy.as_ivy_dtype(
            np_frontend.promote_numpy_dtypes(i.dtype, out_dtype)
        )
    return ivy.stack(arrays, axis=axis, out=out).astype(out_dtype, copy=False)


@to_ivy_arrays_and_back
def vstack(tup):
    out_dtype = ivy.dtype(tup[0])
    for i in tup:
        out_dtype = ivy.as_ivy_dtype(
            np_frontend.promote_numpy_dtypes(i.dtype, out_dtype)
        )
    return ivy.vstack(tup)


row_stack = vstack


@to_ivy_arrays_and_back
def hstack(tup):
    out_dtype = ivy.dtype(tup[0])
    for i in tup:
        out_dtype = ivy.as_ivy_dtype(
            np_frontend.promote_numpy_dtypes(i.dtype, out_dtype)
        )
    return ivy.hstack(tup)
