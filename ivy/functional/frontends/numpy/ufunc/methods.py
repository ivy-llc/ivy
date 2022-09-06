import ivy

from numpy import ufunc, dtype
from typing import Union


@ivy.func_wrapper.outputs_to_ivy_arrays
def accumulate(method: ufunc, array: ivy.Array,
    axis: Union[int, tuple] = 0, dtype: Union[str, dtype] = None) -> ivy.Array:
    return array.__array_ufunc__(method, "accumulate", array=array,
        axis=axis, dtype=dtype)
