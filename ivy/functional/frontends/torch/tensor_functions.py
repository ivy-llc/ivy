# local
import ivy
from ivy.functional.frontends.torch.tensor import Tensor


# TODO: Once the PyTorch Frontend Array Decorators are added,
#  casting to Tensor before returning should be removed as redundant.


def is_tensor(obj):
    return Tensor(ivy.is_array(obj))


# def is_storage(obj):
# 	return ivy.is_storage(obj)

# def is_complex(obj):
# 	return ivy.is_complex(obj)

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
