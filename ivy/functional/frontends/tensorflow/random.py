# global
import ivy


def shuffle(value, seed, name=None):
    return ivy.shuffle(value)


shuffle.unsupported_dtypes = ("float16", "bfloat16")
