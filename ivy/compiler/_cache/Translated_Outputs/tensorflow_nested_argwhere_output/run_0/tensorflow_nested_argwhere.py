from typing import Optional
from typing import Iterable
from typing import Callable
from typing import List
from collections import UserDict
from typing import Union
from typing import Tuple

from .tensorflow__helpers import tensorflow_default


def tensorflow_nested_argwhere(
    nest: Iterable,
    fn: Callable,
    check_nests: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    _index: Optional[List] = None,
    _base: bool = True,
    stop_after_n_found: Optional[int] = None,
):
    to_ignore = tensorflow_default(to_ignore, ())
    _index = [] if _index is None else _index
    if isinstance(nest, (tuple, list)) and not isinstance(nest, to_ignore):
        n = 0
        _indices = []
        for i, item in enumerate(nest):
            ind = (
                tensorflow_nested_argwhere(
                    item,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [i],
                    False,
                    stop_after_n_found - n,
                )
                if stop_after_n_found is not None
                else tensorflow_nested_argwhere(
                    item, fn, check_nests, to_ignore, _index + [i], False, None
                )
            )
            if stop_after_n_found is not None and ind:
                if n >= stop_after_n_found:
                    break
                n = n + len(ind)
            _indices = _indices + [ind]
            if stop_after_n_found is not None and n >= stop_after_n_found:
                break
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    elif isinstance(nest, (dict, UserDict)) and not isinstance(nest, to_ignore):
        n = 0
        _indices = []
        for k, v in nest.items():
            ind = (
                tensorflow_nested_argwhere(
                    v,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [k],
                    False,
                    stop_after_n_found - n,
                )
                if stop_after_n_found is not None
                else tensorflow_nested_argwhere(
                    v, fn, check_nests, to_ignore, _index + [k], False, None
                )
            )
            if stop_after_n_found is not None and ind:
                if n >= stop_after_n_found:
                    break
                n = n + len(ind)
            _indices = _indices + [ind]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    else:
        cond_met = fn(nest)
        if cond_met:
            return [_index]
        return False
    return [index for index in _indices if index]
