# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def is_tensor(obj):
    return ivy.is_array(obj)


# def is_storage(obj):
# 	return ivy.is_storage(obj)

# def is_complex(obj):
# 	return ivy.is_complex(obj)

# def is_conj(obj):
# 	return ivy.is_conj(obj)


@to_ivy_arrays_and_back
def numel(input):
    return input.size


@to_ivy_arrays_and_back
def is_floating_point(input):
    return ivy.is_float_dtype(input)


@to_ivy_arrays_and_back
def is_nonzero(input):
    return ivy.nonzero(input)[0].size != 0
