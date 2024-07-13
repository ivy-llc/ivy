# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def broadcast_tensors(*tensors):
    return ivy.broadcast_arrays(*tensors)


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
    {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
)
def scatter(input, dim, index, src):
    return ivy.put_along_axis(input, index, src, dim, mode="replace")


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
)
def scatter_add(input, dim, index, src):
    return ivy.put_along_axis(input, index, src, dim, mode="sum")


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
)
def scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
    mode_mappings = {
        "sum": "sum",
        "amin": "min",
        "amax": "max",
        "prod": "mul",
        "replace": "replace",
    }
    reduce = mode_mappings.get(reduce, reduce)
    return ivy.put_along_axis(input, index, src, dim, mode=reduce)
