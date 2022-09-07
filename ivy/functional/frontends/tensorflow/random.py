# global
import ivy


def shuffle(value, seed, name=None):
    return ivy.shuffle(value)


shuffle.unsupported_dtypes = ("float16", "bfloat16")


def uniform(shape, minval=0, maxval=None, dtype=None, seed=None, name=None):
    return ivy.random_uniform(low=minval, high=maxval, shape=shape, dtype=dtype)

uniform.unsupported_dtypes = ("float16", "bfloat16")
