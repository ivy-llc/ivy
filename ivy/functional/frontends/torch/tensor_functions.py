# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def is_tensor(obj):
    """
    Checks if the object is a tensor (which is an array)
    Returns True if it a tensor, False otherwise.
    """
    return ivy.is_array(obj)


@to_ivy_arrays_and_back
def numel(input):
    """
    Converts the given array to int64 and returns the same
    """
    return ivy.astype(ivy.array(input.size), ivy.int64)


@to_ivy_arrays_and_back
def is_floating_point(input):
    """
    Checks if the input is of the floating point datatype
    Returns True if yes, False otherwise.
    """
    return ivy.is_float_dtype(input)


@to_ivy_arrays_and_back
def is_nonzero(input):
    """
    Checks if the size of the input is Zero or non zero
    Returns True is input has some value (i.e. is non zero)
    """
    return ivy.nonzero(input)[0].size != 0


@to_ivy_arrays_and_back
def is_complex(input):
    """
    Checks if the input is of complex nature
    Returns True if complex. False otherwise.
    """
    return ivy.is_complex_dtype(input)
