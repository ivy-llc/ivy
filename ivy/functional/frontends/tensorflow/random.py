import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
    to_ivy_dtype,
)


@with_supported_dtypes({"2.9.0 and below": ("float16", "bfloat16", "float32", "float64"," int32", "int64",)}, "tensorflow")
@to_ivy_arrays_and_back
def uniform(
    shape,
    minval=0,
    maxval=None,
    dtype=ivy.float32,
    seed=None,
    name=None
):
    if str(dtype) == "<dtype: 'float32'>": # added this as currently to_ivy_dtype is unable to handed tf.float32 
        dtype = ivy.float32
    return ivy.random_uniform(low=minval, high=maxval, shape=shape, device=ivy.default_device(), dtype=to_ivy_dtype(dtype), seed=seed)