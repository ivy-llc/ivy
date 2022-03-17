"""
Collection of general Ivy functions.
"""

# global
import gc
from typing import Callable, Any, Union, List, Tuple, Dict, Iterable, Optional

#local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

def get_referrers_recursive(item, depth=0, max_depth=None, seen_set=None, local_set=None):
    seen_set = ivy.default(seen_set, set())
    local_set = ivy.default(local_set, set())
    ret_cont = ivy.Container(
        repr=str(item).replace(' ', ''), alphabetical_keys=False, keyword_color_dict={'repr': 'magenta'})
    referrers = [ref for ref in gc.get_referrers(item) if
                 not (isinstance(ref, dict) and
                      min([k in ref for k in ['depth', 'max_depth', 'seen_set', 'local_set']]))]
    local_set.add(str(id(referrers)))
    for ref in referrers:
        ref_id = str(id(ref))
        if ref_id in local_set or hasattr(ref, 'cell_contents'):
            continue
        seen = ref_id in seen_set
        seen_set.add(ref_id)
        refs_rec = lambda: get_referrers_recursive(ref, depth + 1, max_depth, seen_set, local_set)
        this_repr = 'tracked' if seen else str(ref).replace(' ', '')
        if not seen and (not max_depth or depth < max_depth):
            val = ivy.Container(
                repr=this_repr, alphabetical_keys=False, keyword_color_dict={'repr': 'magenta'})
            refs = refs_rec()
            for k, v in refs.items():
                val[k] = v
        else:
            val = this_repr
        ret_cont[str(ref_id)] = val
    return ret_cont


def is_array(x: Any, exclusive: bool = False)\
        -> bool:
    """
    Determines whether the input x is an Ivy Array.

    :param x: The input to check
    :type x: any
    :param exclusive: Whether to check if the data type is exclusively an array, rather than a variable or traced array.
    :type exclusive: bool, optional
    :return: Boolean, whether or not x is an array.
    """
    try:
        return _cur_framework(x).is_array(x, exclusive)
    except ValueError:
        return False


# noinspection PyShadowingNames
def copy_array(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Copy an array.

    :param x: The array to copy
    :type x: array
    :return: A copy of the input array.
    """
    return _cur_framework(x).copy_array(x)


def array_equal(x0: Union[ivy.Array, ivy.NativeArray], x1: Union[ivy.Array, ivy.NativeArray])\
        -> bool:
    """
    Determines whether two input arrays are equal across all elements.

    :param x0: The first input array to compare.
    :type x0: array
    :param x1: The second input array to compare.
    :type x1: array
    :return: Boolean, whether or not the input arrays are equal across all elements.
    """
    return _cur_framework(x0).array_equal(x0, x1)


def arrays_equal(xs: List[Union[ivy.Array, ivy.NativeArray]])\
        -> bool:
    """
    Determines whether input arrays are equal across all elements.

    :param xs: Sequence of arrays to compare for equality
    :type xs: sequence of arrays
    :return: Boolean, whether or not all of the input arrays are equal across all elements.
    """
    x0 = xs[0]
    for x in xs[1:]:
        if not array_equal(x0, x):
            return False
    return True


def all_equal(*xs: Iterable[Any], equality_matrix: bool = False)\
        -> Union[bool, Union[ivy.Array, ivy.NativeArray]]:
    """
    Determines whether the inputs are all equal.

    :param xs: inputs to compare.
    :type xs: any
    :param equality_matrix: Whether to return a matrix of equalities comparing each input with every other.
                            Default is False.
    :type equality_matrix: bool, optional
    :return: Boolean, whether or not the inputs are equal, or matrix array of booleans if equality_matrix=True is set.
    """
    equality_fn = ivy.array_equal if ivy.is_array(xs[0]) else lambda a, b: a == b
    if equality_matrix:
        num_arrays = len(xs)
        mat = [[None for _ in range(num_arrays)] for _ in range(num_arrays)]
        for i, xa in enumerate(xs):
            for j_, xb in enumerate(xs[i:]):
                j = j_ + i
                res = equality_fn(xa, xb)
                if ivy.is_array(res):
                    # noinspection PyTypeChecker
                    res = ivy.to_scalar(res)
                # noinspection PyTypeChecker
                mat[i][j] = res
                # noinspection PyTypeChecker
                mat[j][i] = res
        return ivy.array(mat)
    x0 = xs[0]
    for x in xs[1:]:
        if not equality_fn(x0, x):
            return False
    return True