import ivy
from ivy import Optional, Union, Tuple, List

"""
Parameters
    ----------
    x:
        input array. Should have a numeric data type.
    axes:
        the axes along which to perform the op.
    dtype:
        array data type.
    dev:
        the device on which to place the new array.
"""


def unique_inverse(x: Union[ivy.Array, ivy.NativeArray],
                   axes: Union[int, Tuple[int], List[int]],
                   dtype: Optional[Union[ivy.Dtype, str]] = None,
                   dev: Optional[Union[ivy.Dev, str]] = None) \
        -> ivy.Tuple[ivy.Array, ivy.Array]:

    return _cur_framework(x).my_func(x, dtype, dev)
