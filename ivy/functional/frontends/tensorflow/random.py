import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


@with_unsupported_dtypes(
    {"2.9.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def uniform(shape, minval=0, maxval=None, dtype=ivy.float32, seed=None, name=None):
    return ivy.random_uniform(
        shape=shape, low=minval, high=maxval, dtype=dtype, seed=seed
    )


@with_unsupported_dtypes(
    {"2.9.0 and below": ("int8", "int16", "int32", "int64", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def normal(shape, mean=0.0, stddev=1.0, dtype=ivy.float32, seed=None, name=None):
    return ivy.random_normal(mean=mean, std=stddev, shape=shape, dtype=dtype, seed=seed)


# implement random shuffle
@with_unsupported_dtypes(
    {"2.9.0 and below": ("int8", "int16", "in32", "int64", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def normal(value, seed=None, name=None):
    return ivy.shuffle(x=value, seed=seed, name=name)

