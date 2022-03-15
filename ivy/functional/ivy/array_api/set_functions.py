# global
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

"""
Parameters
    ----------
    x:
        input array. Should have a numeric data type.
"""


def unique_inverse(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Tuple[ivy.Array, ivy.Array]:

    return _cur_framework.unique_inverse(x)
