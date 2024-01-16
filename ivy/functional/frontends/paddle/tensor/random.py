# global
from ..random import *  # noqa: F401
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)

# NOTE:
# Only inplace functions are to be added in this file.
# Please add non-inplace counterparts to `/frontends/paddle/random.py`.


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def exponential_(x, lam=1.0, name=None):
    return ivy.multiply(lam, ivy.exp(ivy.multiply(-lam, x)))


@with_supported_dtypes(
    {"2.5.0 and below": ("int32", "int64", "float32", "float64")},
    "paddle",
)
def randperm(n, dtype=ivy.int64, name=None):
    arr = ivy.arange(n, dtype=dtype)
    ret = ivy.shuffle(arr)
    return ret


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def uniform_(x, min=-1.0, max=1.0, seed=0, name=None):
    x = ivy.array(x)
    return ivy.random_uniform(
        low=min, high=max, shape=x.shape, dtype=x.dtype, seed=seed
    )
