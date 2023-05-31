# local
from collections import namedtuple
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def insert(arr, obj, values, axis=None):
    arr = ivy.array(arr)
    values = ivy.array(values)
    shape = ivy.shape(arr)
    ndim = len(ivy.shape(arr))
    if axis is None:
        arr = ivy.flatten(arr)
        ndim = len(ivy.shape(arr))
        axis = ndim - 1
        shape = ivy.shape(arr)
    elif axis < 0:
        axis += ndim
    if isinstance(obj, slice):
        indices = ivy.arange(*obj.indices(shape[axis]), dtype=ivy.int32)
    else:
        obj = [obj]
        indices = ivy.array(obj).astype(ivy.int32)
    if len(indices) == 0:
        return arr
    elif len(indices) == 1:
        index = int(indices[0])
        if index < -shape[axis] or index > shape[axis]:
            raise IndexError(f"index {obj} is out of bounds for axis {axis} "
                             f"with size {shape[axis]}")
        if index < 0:
            index += shape[axis]
        values = ivy.reshape(values, [-1] + [1] * (ndim - 1))
        values = ivy.moveaxis(values, 0, axis)
        numnew = ivy.shape(values)[axis]
        newshape = list(shape)
        newshape[axis] += numnew
        new = ivy.empty(newshape, dtype=arr.dtype).astype(arr.dtype)
        slobj = [slice(None)] * ndim
        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(index, index + numnew)
        new[tuple(slobj)] = values
        slobj[axis] = slice(index + numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]
        return new

    else:
        order = ivy.argsort(indices, kind='mergesort')
        indices[order] += ivy.arange(len(indices))
        old_mask = ivy.ones(shape[axis], dtype=bool)
        old_mask[indices] = False
        numnew = len(indices)
        newshape = list(shape)
        newshape[axis] += numnew
        new = ivy.empty(newshape, dtype=arr.dtype)
        slobj = [slice(None)] * ndim
        slobj[axis] = indices
        new[tuple(slobj)] = values
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = old_mask
        new[tuple(slobj2)] = arr
        return new


@to_ivy_arrays_and_back
def unique(
    array, /, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    results = ivy.unique_all(array, axis=axis)

    fields = ["values"]
    if return_index:
        fields.append("indices")
    if return_inverse:
        fields.append("inverse_indices")
    if return_counts:
        fields.append("counts")

    Results = namedtuple("Results", fields)

    values = [results.values]
    if return_index:
        values.append(results.indices)
    if return_inverse:
        values.append(results.inverse_indices)
    if return_counts:
        values.append(results.counts)

    return Results(*values)


@to_ivy_arrays_and_back
def append(arr, values, axis=None):
    if axis is None:
        return ivy.concat((ivy.flatten(arr), ivy.flatten(values)), axis=0)
    else:
        return ivy.concat((arr, values), axis=axis)


@to_ivy_arrays_and_back
def trim_zeros(filt, trim="fb"):
    first = 0
    trim = trim.upper()
    if "F" in trim:
        for i in filt:
            if i != 0.0:
                break
            else:
                first = first + 1
    last = len(filt)
    if "B" in trim:
        for i in filt[::-1]:
            if i != 0.0:
                break
            else:
                last = last - 1
    return filt[first:last]
