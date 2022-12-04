import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
)


@handle_numpy_casting
@to_ivy_arrays_and_back
def sorting_comple(array, dtype=numeric):
    if dtype:
        array = [ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype)) for a in array]
    return ivy.sort_complex(array, dtype=numeric)


@to_ivy_arrays_and_back
def sort_complex(array):
    if len(array) == 1:
        return array
    else:
        ivy.sort(array)
        return ivy.sort_complex(array)
