"""
Collection of Ivy functions for nested objects.
"""

# global
from builtins import map as _map
from typing import Callable, Any, Union, List, Dict, Iterable

# local
import ivy


def index_nest(nest, index):
    """
    Index a nested object, using a tuple of indices or keys in the case of dicts.

    :param nest: The nested object to index.
    :type nest: nested
    :param index: A tuple of indices for indexing.
    :type index: tuple of indices
    """
    ret = nest
    for i in index:
        ret = ret[i]
    return ret


def set_nest_at_index(nest, index, value):
    """
    Set the value of a nested item at a specified index

    :param nest: The nested object to update.
    :type nest: nested
    :param index: A tuple of indices for the index at which to update.
    :type index: tuple of indices
    :param value: The new value for updating.
    :type value: any
    """
    if len(index) == 1:
        nest[index[0]] = value
    else:
        set_nest_at_index(nest[index[0]], index[1:], value)


def map_nest_at_index(nest, index, fn):
    """
    Map a function to the value of a nested item at a specified index

    :param nest: The nested object to update.
    :type nest: nested
    :param index: A tuple of indices for the index at which to update.
    :type index: tuple of indices
    :param fn: The function to perform on the nest at the given index.
    :type fn: callable
    """
    if len(index) == 1:
        nest[index[0]] = fn(nest[index[0]])
    else:
        map_nest_at_index(nest[index[0]], index[1:], fn)


def multi_index_nest(nest, indices):
    """
    Repeatedly index a nested object, using a tuple of tuples of indices or keys in the case of dicts.

    :param nest: The nested object to slice.
    :type nest: nested
    :param indices: A tuple of tuples of indices to apply.
    :type indices: tuple of tuples of indices
    """
    return [index_nest(nest, index) for index in indices]


def set_nest_at_indices(nest, indices, values):
    """
    Set the value of a nested item at specified indices with specified values.

    :param nest: The nested object to update.
    :type nest: nested
    :param indices: A tuple of tuples of indices for the indices at which to update.
    :type indices: tuple of tuples of indices
    :param values: The new values for updating.
    :type values: sequence of any
    """
    if not isinstance(values, (list, tuple)):
        values = [values]*len(indices)
    [set_nest_at_index(nest, index, value) for index, value in zip(indices, values)]


def map_nest_at_indices(nest, indices, fn):
    """
    Map a function to the values of a nested item at the specified indices

    :param nest: The nested object to update.
    :type nest: nested
    :param indices: A tuple of tuples of indices for the indices at which to update.
    :type indices: tuple of tuples of indices
    :param fn: The function to perform on the nest at the given index.
    :type fn: callable
    """
    [map_nest_at_index(nest, index, fn) for index in indices]


def nested_indices_where(nest: Iterable, fn: Callable, check_nests: bool = False, _index: List = None,
                         _base: bool = True) -> Union[Iterable, bool]:
    """
    Checks the leaf nodes of nested x via function fn, and returns all nest indices where the method evaluates as True.

    :param nest: The nest to check the leaves of.
    :type nest: nest of any
    :param fn: The conditon function, returning True or False.
    :type fn: callable
    :param check_nests: Whether to also check the nests for the condition, not only nest leaves. Default is False.
    :type check_nests: bool, optional
    :param _index: The indices detected so far. None at the beginning. Used internally, do not set manually.
    :type _index: list of tuples of indices, do not set
    :param _base: Whether the current function call is the first function call in the recursive stack.
                  Used internally, do not set manually.
    :type _base: bool, do not set
    :return: A set of indices for the nest where the function evaluated as True.
    """
    _index = list() if _index is None else _index
    if isinstance(nest, (tuple, list)):
        _indices = [nested_indices_where(item, fn, check_nests, _index + [i], False) for i, item in enumerate(nest)]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    elif isinstance(nest, dict):
        _indices = [nested_indices_where(v, fn, check_nests, _index + [k], False) for k, v in nest.items()]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    else:
        cond_met = fn(nest)
        if cond_met:
            return [_index]
        return False
    return [index for index in _indices if index]


def all_nested_indices(nest: Iterable, include_nests: bool = False, _index: List = None, _base: bool = True)\
        -> Union[Iterable, bool]:
    """
    Checks the leaf nodes of nested x via function fn, and returns all nest indices where the method evaluates as True.

    :param nest: The nest to check the leaves of.
    :type nest: nest of any
    :param include_nests: Whether to also include indices of the nests themselves, not only leaves. Default is False.
    :type include_nests: bool, optional
    :param _index: The indices detected so far. None at the beginning. Used internally, do not set manually.
    :type _index: list of tuples of indices, do not set
    :param _base: Whether the current function call is the first function call in the recursive stack.
                  Used internally, do not set manually.
    :type _base: bool, do not set
    :return: A set of indices for the nest where the function evaluated as True.
    """
    _index = list() if _index is None else _index
    if isinstance(nest, (tuple, list)):
        _indices = [all_nested_indices(item, include_nests, _index + [i], False) for i, item in enumerate(nest)]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if include_nests:
            _indices.append(_index)
    elif isinstance(nest, dict):
        _indices = [all_nested_indices(v, include_nests, _index + [k], False) for k, v in nest.items()]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if include_nests:
            _indices.append(_index)
    else:
        return [_index]
    return [index for index in _indices if index]


# noinspection PyShadowingBuiltins
def map(fn: Callable, constant: Dict[str, Any] = None, unique: Dict[str, Iterable[Any]] = None, mean: bool = False)\
        -> List:
    """
    Applies a function on each item of an iterable x.

    :param fn: The function to map onto x.
    :type fn: callable
    :param constant: keyword arguments which remain constant between each function call. Default is None.
    :type constant: dict of any, optional
    :param unique: keyword arguments which are unique for each function call. Default is None.
    :type unique: dict of iterables of any, optional
    :param mean: Whether to compute the mean across the return values, and return this mean. Default is False.
    :type mean: bool, optional
    :return: x following the applicable of fn to each of it's iterated items.
    """
    c = ivy.default(constant, {})
    u = ivy.default(unique, {})
    rets = [r for r in _map(lambda *uv: fn(**dict(**c, **dict(zip(u.keys(), uv)))), *u.values())]
    if mean:
        return sum(rets) / len(rets)
    return rets


def nested_map(x: Union[Union[ivy.Array, ivy.NativeArray], Iterable], fn: Callable, include_derived: bool = False,
               to_mutable: bool = False, max_depth: int = None, depth: int = 0)\
        -> Union[Union[ivy.Array, ivy.NativeArray], Iterable]:
    """
    Applies a function on x in a nested manner, whereby all dicts, lists and tuples are traversed to their lowest
    leaves before applying the method and returning x. If x is not nested, the method is applied to x directly.

    :param x: The item to apply the mapped function to.
    :type x: any
    :param fn: The function to map onto x.
    :type fn: callable
    :param include_derived: Whether to also recursive for classes derived from tuple, list and dict. Default is False.
    :type include_derived: bool, optional
    :param to_mutable: Whether to convert the nest to a mutable form, changing all tuples to lists. Default is False.
    :type to_mutable: bool, optional
    :param max_depth: The maximum nested depth to reach. Default is 1. Increase this if the nest is deeper.
    :type max_depth: int, optional
    :param depth: Placeholder for tracking the recursive depth, do not yet this parameter.
    :type depth: int, used internally
    :return: x following the applicable of fn to it's nested leaves, or x itself if x is not nested.
    """
    if ivy.exists(max_depth) and depth > max_depth:
        return x
    class_instance = type(x)
    check_fn = (lambda x_, t: isinstance(x, t)) if include_derived else (lambda x_, t: type(x) is t)
    if check_fn(x, tuple):
        ret_list = [nested_map(i, fn, include_derived, to_mutable, max_depth, depth + 1) for i in x]
        if to_mutable:
            return ret_list
        return class_instance(tuple(ret_list))
    elif check_fn(x, list):
        return class_instance([nested_map(i, fn, include_derived, to_mutable, max_depth, depth+1) for i in x])
    elif check_fn(x, dict):
        class_instance = type(x)
        return class_instance({k: nested_map(v, fn, include_derived, to_mutable, max_depth, depth+1)
                               for k, v in x.items()})
    return fn(x)


def copy_nest(nest: Union[Union[ivy.Array, ivy.NativeArray], Iterable], include_derived: bool = False,
              to_mutable: bool = False)\
        -> Union[Union[ivy.Array, ivy.NativeArray], Iterable]:
    """
    Copies a nest deeply, but without copying leaves of the nest, only the nest lists, tuples and dicts are copied.

    :param nest: The nest to copy.
    :type nest: nested
    :param include_derived: Whether to also recursive for classes derived from tuple, list and dict. Default is False.
    :type include_derived: bool, optional
    :param to_mutable: Whether to convert the nest to a mutable form, changing all tuples to lists. Default is False.
    :type to_mutable: bool, optional
    :return: The copied nest.
    """
    class_instance = type(nest)
    check_fn = (lambda x_, t: isinstance(nest, t)) if include_derived else (lambda x_, t: type(nest) is t)
    if check_fn(nest, tuple):
        ret_list = [copy_nest(i, include_derived, to_mutable) for i in nest]
        if to_mutable:
            return ret_list
        return class_instance(tuple(ret_list))
    elif check_fn(nest, list):
        return class_instance([copy_nest(i, include_derived, to_mutable) for i in nest])
    elif check_fn(nest, dict):
        class_instance = type(nest)
        return class_instance({k: copy_nest(v, include_derived, to_mutable) for k, v in nest.items()})
    return nest
