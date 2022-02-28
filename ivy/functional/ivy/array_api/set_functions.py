# global
from typing import Union, Tuple

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

def unique_counts(x: Union[ivy.Array, ivy.NativeArray]) \
                -> Tuple[ivy.Array, ivy.Array]:
    """
    Returns the unique elements of an input array x and the corresponding counts for each unique element in x.

    :param x: input array. If x has more than one dimension, the function must flatten x and return the unique elements of the flattened array.
    :return: a namedtuple (values, counts) whose
            -first element must have the field name values and must be an array containing the unique elements of x. The array must have the same data type as x.
            -second element must have the field name counts and must be an array containing the number of times each unique element occurs in x. The returned array must have same shape as values and must have the default array index data type.
    """
    return _cur_framework(x).unique_counts(x)