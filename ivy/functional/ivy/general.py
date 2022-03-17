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