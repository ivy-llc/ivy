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


@to_ivy_arrays_and_back
def rollaxis(a, axis, start=0):
    n = len(ivy.shape(a))
    if axis < -n or axis >= n:
        raise ValueError(f"axis {axis} is out of bounds for array of {n} dimensions")
    if axis < 0:
        axis += n
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise ValueError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        start -= 1
    end = start + axis
    axes = tuple(i for i in range(n) if i != axis)
    axes = axes[:start] + (axis,) + axes[start:end] + axes[end:]
    return ivy.permute_dims(a, axes, out=None)
