# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_jax_dtype,
)


@to_ivy_arrays_and_back
def PRNGKey(seed):
    return ivy.array([0, seed % 4294967295 - (seed // 4294967295)], dtype=ivy.int64)


@handle_jax_dtype
@to_ivy_arrays_and_back
def uniform(key, shape=(), dtype=None, minval=0.0, maxval=1.0):
    return ivy.random_uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, seed=ivy.to_scalar(key[1])
    )


@handle_jax_dtype
@to_ivy_arrays_and_back
def normal(key, shape=(), dtype=None):
    return ivy.random_normal(shape=shape, dtype=dtype, seed=ivy.to_scalar(key[1]))


def _get_seed(key):
    key1, key2 = int(key[0]), int(key[1])
    return ivy.to_scalar(int("".join(map(str, [key1, key2]))))


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def beta(key, a, b, shape=None, dtype=None):
    seed = _get_seed(key)
    return ivy.beta(a, b, shape=shape, dtype=dtype, seed=seed)
