
# global
from typing import Union, Tuple

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


from typing import Union, Tuple


def zeros(shape: Union[int, Tuple[int, ...]], dtype: ivy.Dtype = None, device: ivy.Device = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Return a new array of given shape and type, filled with zeros.

    :param shape: output array shape.
    :param dtype: output array data type. If dtype is None, the output array data type must be the default
                  floating-point data type. Default: None.
    :param device: device on which to place the created array. Default: None.
    :return: an array containing zeros.
    """
    return _cur_framework().zeros(shape, dtype, device)
