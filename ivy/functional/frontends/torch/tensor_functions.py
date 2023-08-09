# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def is_tensor(obj):
    return ivy.is_array(obj)


@to_ivy_arrays_and_back
def numel(input):
    return ivy.astype(ivy.array(input.size), ivy.int64)


@to_ivy_arrays_and_back
def is_floating_point(input):
    return ivy.is_float_dtype(input)


@to_ivy_arrays_and_back
def is_nonzero(input):
    return ivy.nonzero(input)[0].size != 0


@to_ivy_arrays_and_back
def is_complex(input):
    return ivy.is_complex_dtype(input)


@to_ivy_arrays_and_back
def scatter(input, dim, index, src):
    return ivy.put_along_axis(input, index, src, dim)


@to_ivy_arrays_and_back
def scatter_add(input, dim, index, src):
    return ivy.put_along_axis(input, index, src, dim)


@to_ivy_arrays_and_back
def scatter_reduce(input, dim, index, src, reduce=None):
    return ivy.put_along_axis(input, index, src, dim, mode=reduce)
