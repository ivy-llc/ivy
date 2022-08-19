# global
import ivy


def shuffle(value, seed, name=None):
    return ivy.shuffle(value)


# shuffle.unsupported_dtypes = ("float16", "bfloat16")


def uniform(shape, minval, maxval, dtype=None, seed=None, name=None):
    return ivy.random_uniform(shape=shape, low=minval, high=maxval)
