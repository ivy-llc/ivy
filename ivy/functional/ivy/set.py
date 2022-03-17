# global
from typing import Union, Tuple

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def unique_inverse(x: Union[ivy.Array, ivy.NativeArray]) \
        -> Tuple[ivy.Array, ivy.Array]:
    """
    Returns a tuple of two arrays, one being the unique elements of an input array x and the other one the indices from
    the set of uniques elements that reconstruct x.

    :param x: input array.
    :return: tuple of two arrays (values, inverse_indices)
    """
    return _cur_framework.unique_inverse(x)
