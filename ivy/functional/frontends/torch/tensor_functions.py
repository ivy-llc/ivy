# local
from numpy import complex128, complex192, complex64
import ivy
from ivy.functional.frontends.torch.Tensor import Tensor
from ivy.functional.ivy.general import is_array



# TODO: Once the PyTorch Frontend Array Decorators are added,
#  casting to Tensor before returning should be removed as redundant.


def is_tensor(obj):
    return Tensor(ivy.is_array(obj))


# def is_storage(obj):
# 	return ivy.is_storage(obj)

def is_complex(obj):
    if ivy.dtype(obj) == complex64 or ivy.dtype(obj) == complex128 or ivy.dtype(obj) == complex192:
        return Tensor(is_tensor(obj))

    

# def is_conj(obj):
# 	return ivy.is_conj(obj)


def numel(input):
    ivy.assertions.check_true(
        is_tensor(input),
        message="input must be a tensor",
    )
    return Tensor(input.size)


def is_floating_point(input):
    ivy.assertions.check_true(
        is_tensor(input),
        message="input must be a tensor",
    )
    return Tensor(ivy.is_float_dtype(input))

def is_nonzero(input):
    ivy.assertions.check_equal(
        numel(input).data,
        1,
        message="bool value of tensor with more than one or no values is ambiguous",
    )
    return Tensor(ivy.nonzero(input)[0].size != 0)
