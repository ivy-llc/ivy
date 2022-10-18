# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def transpose(array, /, *, axes=None):
    if axes is None:
        axes = list(range(len(array.shape)))[::-1]
    assert len(axes) > 1, "`axes` should have the same size the input array.ndim"

    return ivy.permute_dims(array, axes, out=None)


@to_ivy_arrays_and_back
def swapaxes(a, axis1, axis2):
    return ivy.swapaxes(a, axis1, axis2)
