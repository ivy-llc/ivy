from ivy.core.container import Container
from ivy.core.operations import arithmetic_ops
from ivy.core.tensor import Tensor
from ivy.numpy.base import promote_types_of_numpy_inputs


def percentile(arr, q, axis=None, out=None, overwrite_input=False, interpolation='linear'):
    arr = promote_types_of_numpy_inputs(arr)
    if axis is None:
        arr = arr.flatten()
        axis = 0
    elif axis < 0:
        axis += arr.ndim
    if out is not None:
        out = promote_types_of_numpy_inputs(out)

   
    sorted_arr = arithmetic_ops.sort(arr, axis=axis)

    
    alpha = q / 100.0
    if interpolation == 'linear':
        index = alpha * (sorted_arr.shape[axis] - 1)
        lower_idx = arithmetic_ops.floor(index)
        upper_idx = arithmetic_ops.ceil(index)
        weight = index - lower_idx
    elif interpolation == 'lower':
        index = alpha * sorted_arr.shape[axis]
        lower_idx = arithmetic_ops.floor(index)
        upper_idx = lower_idx
        weight = 0
    elif interpolation == 'higher':
        index = alpha * sorted_arr.shape[axis]
        lower_idx = arithmetic_ops.ceil(index)
        upper_idx = lower_idx
        weight = 0
    else:
        raise ValueError("Interpolation method '{}' not supported.".format(interpolation))


    if out is not None:
        ret = out
    else:
        ret = Container(arr)
    ret.val = (1.0 - weight) * sorted_arr.take(lower_idx, axis=axis) + weight * sorted_arr.take(upper_idx, axis=axis)

    if overwrite_input:
        arr.val = ret.val
        return arr
    else:
        return ret
