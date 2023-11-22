# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def argsort(
    x,
    /,
    *,
    axis=-1,
    kind=None,
    order=None,
):
    return ivy.argsort(x, axis=axis)


@to_ivy_arrays_and_back
def lexsort(keys, /, *, axis=-1):
    return ivy.lexsort(keys, axis=axis)


@to_ivy_arrays_and_back
def msort(a):
    return ivy.msort(a)


@to_ivy_arrays_and_back
def partition(a, kth, axis=-1, kind="introselect", order=None):
    sorted_arr = ivy.sort(a, axis=axis)
    for k in kth:
        index_to_remove = ivy.argwhere(a == sorted_arr[k])[0, 0]
        if len(a) == 1:
            a = ivy.array([], dtype=a.dtype)
        else:
            a = ivy.concat((a[:index_to_remove], a[index_to_remove + 1 :]))
        left = ivy.array([], dtype=a.dtype)
        right = ivy.array([], dtype=a.dtype)
        equal = ivy.array([], dtype=a.dtype)
        for i in range(len(a)):
            if a[i] < sorted_arr[k]:
                left = ivy.concat((left, ivy.array([a[i]], dtype=a.dtype)))
            elif a[i] > sorted_arr[k]:
                right = ivy.concat((right, ivy.array([a[i]], dtype=a.dtype)))
            else:
                equal = ivy.concat((equal, ivy.array([a[i]], dtype=a.dtype)))
        for j in range(len(equal)):
            if len(left) == len(sorted_arr[:k]):
                right = ivy.concat((right, ivy.array([equal[j]], dtype=a.dtype)))
            else:
                left = ivy.concat((left, ivy.array([equal[j]], dtype=a.dtype)))
        a = ivy.concat((left, ivy.array([sorted_arr[k]], dtype=a.dtype), right))
    return a


@to_ivy_arrays_and_back
def sort(a, axis=-1, kind=None, order=None):
    return ivy.sort(a, axis=axis)


@to_ivy_arrays_and_back
def sort_complex(a):
    return ivy.sort(a)
