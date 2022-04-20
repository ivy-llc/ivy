# global
from typing import get_type_hints

# local
import ivy


def _is_optional(typ):
    # noinspection PyBroadException
    try:
        rep = typ.__repr__().split('.')[1]
        if rep.startswith('Optional') or (rep.startswith('Union') and type(None) in typ.__args__):
            return True
    except:
        pass
    return False


def _is_iterable(typ):
    # noinspection PyBroadException
    try:
        rep = typ.__repr__().split('.')[1]
        if rep.startswith('List') or rep.startswith('Tuple') or rep.startswith('Dict') or rep.startswith('Set'):
            return True
    except:
        pass
    return False


def _get_array_idxs(typ, idx_so_far=None):
    idx_so_far = ivy.default(idx_so_far, list())
    these_idxs = list()
    if not hasattr(typ, '__args__'):
        return these_idxs
    is_opt = _is_optional(typ)
    is_iter = _is_iterable(typ)
    for i, a in enumerate(typ.__args__):
        if a is ivy.NativeArray:
            these_idxs.append(idx_so_far + (['optional'] if is_opt else ([int] if is_iter else [])))
        else:
            these_idxs += _get_array_idxs(a, idx_so_far + (['optional'] if is_opt else ([int] if is_iter else [])))
    return these_idxs


def fn_array_spec(fn):
    """
    Return a specification of the function, indicating all arguments which include arrays, and the indexes of these.

    :param fn: function to inspect
    :type fn: callable
    :return: specification
    """
    type_hints = get_type_hints(fn)
    array_idxs = list()
    for i, (k, v) in enumerate(type_hints.items()):
        a_idxs = _get_array_idxs(v)
        if a_idxs:
            a_idxs = [[(i, k)] + a for a in a_idxs]
            array_idxs += a_idxs
    return array_idxs
