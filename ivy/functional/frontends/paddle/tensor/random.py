# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.func_wrapper import with_supported_device_and_dtypes, with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def uniform(shape, dtype=None, min=-1.0, max=1.0, seed=0, name=None):
    return ivy.random_uniform(low=min, high=max, shape=shape, dtype=dtype, seed=seed)


@to_ivy_arrays_and_back
def randint(low=0, high=None, shape=[1], dtype=None, name=None):
    return ivy.randint(low, high, shape=shape, dtype=dtype)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def poisson(x, name=None):
    return ivy.poisson(x, shape=None, device=None, dtype=None, seed=None, out=None)


@with_supported_device_and_dtypes(
    {
        "2.5.0 and above": {
            "cpu": (
                "bfloat16",
                "float32",
                "float64",
            ),
            "gpu": (
                "bfloat16",
                "float16",
                "float32",
                "float64",
            ),
        },
        "2.4.2 and below": {
            "cpu": (
                "float32",
                "float64",
            ),
            "gpu": (
                "float16",
                "float32",
                "float64",
            ),
        },
    },
    "paddle",
)
@to_ivy_arrays_and_back
def rand(shape, dtype=None, name=None):
    return ivy.random_uniform(low=0.0, high=1.0, shape=shape, dtype=dtype, seed=None)


def randn(shape, dtype=None, name=None):
    if dtype not in ["float32", "float64"]:
        raise ivy.exceptions.IvyError(
            "Unsupported dtype for randn, only float32 and float64 are supported, "
        )
    return ivy.random_normal(shape=shape, dtype=dtype, seed=None)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def uniform_(x, min=-1.0, max=1.0, seed=0, name=None):
    x = ivy.array(x)
    return ivy.random_uniform(
        low=min, high=max, shape=x.shape, dtype=x.dtype, seed=seed
    )


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def standard_normal(shape, dtype=None, name=None):
    return ivy.random_normal(mean=0, std=1, shape=shape, dtype=dtype)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("int16", "float16", "bfloat16", "uint8")},
    "paddle",
)
@to_ivy_arrays_and_back
def randint_like(x, low=0, high=None, dtype=None, name=None):
    if high is None:
        high = low
        low = 0
        if high <= 0:
            raise ivy.exceptions.IvyError(
                "If high is None, low must be greater than 0, but received low = 0."
            )
    return ivy.randint(low, high, shape=x.shape, dtype=dtype, seed=None)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def exponential_(x, lam=1.0, name=None):
    x = ivy.array(x)
    uniform = ivy.random_uniform(shape=x.shape, dtype=x.dtype)
    return -lam * ivy.log(lam - uniform)
