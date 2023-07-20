import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    inputs_to_ivy_arrays,
    handle_numpy_out,
)

@to_ivy_arrays_and_back
#ToDo: There is no Ivy implementation, so take() is implemented with Ivy functions
# and Python. This will be simplified when ivy.take() is added to the API
def take(a, indices, axis=None, out=None, mode='raise'):
    a = ivy.array(a)

    # Default: axis None flattens array
    if axis is None:
        a = a.flatten()

    # Handle invalid mode input
    if mode not in ('raise', 'wrap', 'clip'):
        raise ValueError(f"clipmode must be one of 'clip', 'raise', or 'wrap' (got '{mode}')")

    # Returns axis of input array, can be None if axis is too large
    inp_axis = a.shape[axis]

    result = []
    for index in indices:
        # Handles error where axis is larger than highest dimension of array
        if inp_axis is None:
          raise ValueError(f"axis {axis} is out of bounds for array of dimension {len(a.shape)}")
        
        # Raise error for invalid values
        if mode == 'raise':
            if index < 0 or (inp_axis is not None and index >= inp_axis):
                raise IndexError(f"index {index} is out of bounds for axis {axis} of size {inp_axis}")

        # Clips index at last valid index for input index
        if mode == 'clip':
            index = min(max(index, 0), inp_axis - 1)

        # Returns value indexed with remainder of index and axis
        if mode == 'wrap':
            index = index % inp_axis

        if axis is None:
            # Adds to result  from indexed flattened array
            result.append(a[index])
        else:
            slicer = [slice(None)] * len(a.shape)

            # Indexes input array to include values from same axis
            # in slice object
            slicer[axis] = index

            # Adds slice object to output
            result.append(a[tuple(slicer)])

    # Note: Since output is an ivy array, out if not None should be set to ivy array
    if out is not None and out.shape == a.shape and type(out) == type(a):
        out[:] = result
        return out

    return ivy.array(result)


@to_ivy_arrays_and_back
def take_along_axis(arr, indices, axis):
    return ivy.take_along_axis(arr, indices, axis)


@to_ivy_arrays_and_back
def tril_indices(n, k=0, m=None):
    return ivy.tril_indices(n, m, k)


@to_ivy_arrays_and_back
def indices(dimensions, dtype=int, sparse=False):
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = tuple()
    else:
        res = ivy.empty((N,) + dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = ivy.arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i + 1 :])
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


# unravel_index
@to_ivy_arrays_and_back
def unravel_index(indices, shape, order="C"):
    ret = [x.astype("int64") for x in ivy.unravel_index(indices, shape)]
    return tuple(ret)


@to_ivy_arrays_and_back
def fill_diagonal(a, val, wrap=False):
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
        # This is needed to don't have tall matrix have the diagonal wrap.
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not ivy.all(ivy.diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + ivy.sum(ivy.cumprod(a.shape[:-1]))

    # Write the value out into the diagonal.
    shape = a.shape
    a = ivy.reshape(a, a.size)
    a[:end:step] = val
    a = ivy.reshape(a, shape)


@inputs_to_ivy_arrays
def put_along_axis(arr, indices, values, axis):
    ivy.put_along_axis(arr, indices, values, axis)


def diag(v, k=0):
    return ivy.diag(v, k=k)


@to_ivy_arrays_and_back
def diagonal(a, offset, axis1, axis2):
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


@to_ivy_arrays_and_back
@handle_numpy_out
def compress(condition, a, axis=None, out=None):
    condition_arr = ivy.asarray(condition).astype(bool)
    if condition_arr.ndim != 1:
        raise ivy.utils.exceptions.IvyException("Condition must be a 1D array")
    if axis is None:
        arr = ivy.asarray(a).flatten()
        axis = 0
    else:
        arr = ivy.moveaxis(a, axis, 0)
    if condition_arr.shape[0] > arr.shape[0]:
        raise ivy.utils.exceptions.IvyException(
            "Condition contains entries that are out of bounds"
        )
    arr = arr[: condition_arr.shape[0]]
    return ivy.moveaxis(arr[condition_arr], 0, axis)
