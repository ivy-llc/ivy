
# global
from typing import Union, Tuple

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


from typing import Union, Tuple


def zeros(shape: Union[int, Tuple[int, ...]], *, dtype: ivy.Dtype = None, device: ivy.Device = None)\
        -> ivy.Array:
    """
    Return a new array of given shape and type, filled with zeros.
    :param shape: Shape of the new array, e.g. (2, 3).
    :type shape: sequence of ints
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype: data-type string, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev: ivy.Device
    :return: an array containing zeros. .
    """
    return _cur_framework().zeros(shape, dtype, device)