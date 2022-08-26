import ivy
import numpy as np


def sum(
    x,
    /,
    *,
    axis=None,
    dtype=None,
    keepdims=False,
    out=None,
    initial=np._NoValue,
    where=np._NoValue,
):
    return ivy.sum(
        x,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        out=out,
        initial=initial,
        where=where,
    )


sum.unsupported_dtypes = {"torch": ("float16",)}
