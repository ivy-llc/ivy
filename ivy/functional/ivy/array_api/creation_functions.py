# import global
from typing import Union, Optional, Tuple

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[ivy.Dtype] = 'float32',
         device: Optional[ivy.Device] = None) \
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns a new array of given shape and type, filled with ones.

    :param shape: Shape of the new array, e.g. (2, 3).
    :type shape: iterable int or int
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype: data-type string, optional
    :param device: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type device: ivy.Device
    :return: Tensor of ones with the given shape and dtype.
    """
    return _cur_framework().ones(shape, dtype, device)