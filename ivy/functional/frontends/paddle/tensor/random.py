# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def uniform(shape, dtype=None, min=-1.0, max=1.0, seed=0, name=None):
    return ivy.random_uniform(low=min, high=max, shape=shape, dtype=dtype, seed=seed)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def randn(shape, dtype=None, seed=0, name=None):
    return ivy.random_normal(shape=shape, dtype=dtype, seed=seed)
