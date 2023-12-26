# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def is_complex(input):
    return ivy.is_complex_dtype(input)

@to_ivy_arrays_and_back
def is_floating_point(input):
    return ivy.is_float_dtype(input)

@to_ivy_arrays_and_back
def is_nonzero(input):
    return ivy.nonzero(input)[0].size != 0

@to_ivy_arrays_and_back
def is_tensor(obj):
    return ivy.is_array(obj)

@to_ivy_arrays_and_back
def numel(input):
    return ivy.astype(ivy.array(input.size), ivy.int64)

@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "int32", "int64")}, "numpy"
)
def scatter(input, dim, index, src):
    index_shape = [1] * input.ndim
    index_shape[dim] = -1
    index = ivy.unstack(ivy.expand_dims(index, dim), axis=-1)
    src = ivy.reshape(src, index_shape + list(src.shape[1:]))
    return ivy.scatter_nd_add(input, index, src)

@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "int32", "int64")}, "numpy"
)
def scatter_add(input, dim, index, src):
    index_shape = [1] * input.ndim
    index_shape[dim] = -1
    index = ivy.unstack(ivy.expand_dims(index, dim), axis=-1)
    src = ivy.reshape(src, index_shape + list(src.shape[1:]))
    return ivy.scatter_nd_add(input, index, src)

@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "int32", "int64")}, "numpy"
)
def scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
    mode_mappings = {
        "sum": "add",
        "amin": "min",
        "amax": "max",
        "prod": "multiply",
        "replace": "copy",
    }
    reduce = mode_mappings.get(reduce, reduce)

    index_shape = [1] * input.ndim
    index_shape[dim] = -1
    index = ivy.unstack(ivy.expand_dims(index, dim), axis=-1)
    src = ivy.reshape(src, index_shape + list(src.shape[1:]))
    
    scattered = ivy.scatter_nd_add(input, index, src)
    
    if include_self:
        scattered = getattr(ivy, reduce)(scattered, src)
    return scattered
