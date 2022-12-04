import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
)


@handle_numpy_casting
@to_ivy_arrays_and_back
def sort_complex(array, dtype=complex):
    if dtype:
        array = [ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype)) for a in array]
    return ivy.sort_complex(array, dtype=complex)


@to_ivy_arrays_and_back
def merge_sort(array):
    left_array = array[:len(array) // 2]
    right_array = array[len(array) // 2:]

    merge_sort(left_array)
    merge_sort(right_array)

    i = 0  # left_array
    j = 0  # right_array
    k = 0  # keeping track of index
    while i < len(left_array) and j < len(right_array):
        if left_array[i] < right_array[j]:
            array[k] = left_array[i]
            i += 1
            k += 1
        else:
            array[k] = right_array[j]
            j += 1
            k += 1

        while i < len(left_array):
            array[k] = left_array[i]
            i += 1
            k += 1

        while j < len(right_array):
            array[k] = right_array[j]
            j += 1
            k += 1
    return array


@to_ivy_arrays_and_back
def sort_complex(array):
    if len(array) == 1:
        return array
    else:
        merge_sort(array)
        return ivy.sort_complex(array)
