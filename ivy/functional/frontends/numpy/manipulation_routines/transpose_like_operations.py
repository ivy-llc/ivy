# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def transpose(array, /, *, axes=None):
    if not axes:
        axes = list(range(len(array.shape)))[::-1]
    if type(axes) is int:
        axes = [axes]
    if (len(array.shape) == 0 and not axes) or (len(array.shape) == 1 and axes[0] == 0):
        return array
    return ivy.permute_dims(array, axes, out=None)


@to_ivy_arrays_and_back
def swapaxes(a, axis1, axis2):
    return ivy.swapaxes(a, axis1, axis2)
