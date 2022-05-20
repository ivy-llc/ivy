"""Collection of Ivy functions for nested objects."""

# global
from builtins import map as _map
from typing import Callable, Any, Union, List, Tuple, Optional, Dict, Iterable

# local
import ivy


# Extra #
# ------#


def index_nest(nest, index):
    """Index a nested object, using a tuple of indices or keys in the case of dicts.

    Parameters
    ----------
    nest
        The nested object to index.
    index
        A tuple of indices for indexing.

    """
    ret = nest
    for i in index:
        ret = ret[i]
    return ret


def prune_nest_at_index(nest, index):
    """Prune a nested object at a specified index.

    Parameters
    ----------
    nest
        The nested object to prune.
    index
        A tuple of indices for the index at which to prune.

    """
    if len(index) == 1:
        del nest[index[0]]
    else:
        prune_nest_at_index(nest[index[0]], index[1:])


def set_nest_at_index(nest, index, value):
    """Set the value of a nested item at a specified index.

    Parameters
    ----------
    nest
        The nested object to update.
    index
        A tuple of indices for the index at which to update.
    value
        The new value for updating.

    """
    if len(index) == 1:
        nest[index[0]] = value
    else:
        set_nest_at_index(nest[index[0]], index[1:], value)


def insert_into_nest_at_index(nest, index, value):
    if len(index) == 1:
        idx = index[0]
        if isinstance(nest, list):
            nest.insert(idx, value)
        else:
            nest[index[0]] = value
    else:
        insert_into_nest_at_index(nest[index[0]], index[1:], value)


def map_nest_at_index(nest, index, fn):
    """Map a function to the value of a nested item at a specified index.

    Parameters
    ----------
    nest
        The nested object to update.
    index
        A tuple of indices for the index at which to update.
    fn
        The function to perform on the nest at the given index.

    """
    if len(index) == 1:
        nest[index[0]] = fn(nest[index[0]])
    else:
        map_nest_at_index(nest[index[0]], index[1:], fn)


def multi_index_nest(nest, indices):
    """Repeatedly index a nested object, using a tuple of tuples of indices or keys in
    the case of dicts.

    Parameters
    ----------
    nest
        The nested object to slice.
    indices
        A tuple of tuples of indices to apply.

    """
    return [index_nest(nest, index) for index in indices]


def prune_nest_at_indices(nest, indices):
    """Prune a nested object at specified indices.

    Parameters
    ----------
    nest
        The nested object to prune.
    indices
        A tuple of tuples of indices for the indices at which to prune.

    """
    [prune_nest_at_index(nest, index) for index in indices]


def set_nest_at_indices(nest, indices, values):
    """Set the value of a nested item at specified indices with specified values.

    Parameters
    ----------
    nest
        The nested object to update.
    indices
        A tuple of tuples of indices for the indices at which to update.
    values
        The new values for updating.

    """
    if not isinstance(values, (list, tuple)):
        values = [values] * len(indices)
    [set_nest_at_index(nest, index, value) for index, value in zip(indices, values)]


def insert_into_nest_at_indices(nest, indices, values):
    """Insert a value into the nested item at specified indices with specified values.

    Parameters
    ----------
    nest
        The nested object to insert into.
    indices
        A tuple of tuples of indices for the indices at which to insert
        values.
    values
        The new values for inserting.

    """
    if not isinstance(values, (list, tuple)):
        values = [values] * len(indices)
    [
        insert_into_nest_at_index(nest, index, value)
        for index, value in zip(indices, values)
    ]


def map_nest_at_indices(nest, indices, fn):
    """Map a function to the values of a nested item at the specified indices.

    Parameters
    ----------
    nest
        The nested object to update.
    indices
        A tuple of tuples of indices for the indices at which to update.
    fn
        The function to perform on the nest at the given index.

    """
    [map_nest_at_index(nest, index, fn) for index in indices]


def nested_indices_where(
    nest: Iterable,
    fn: Callable,
    check_nests: bool = False,
    to_ignore: Union[type, Tuple[type]] = None,
    _index: List = None,
    _base: bool = True,
) -> Union[Iterable, bool]:
    """Checks the leaf nodes of nested x via function fn, and returns all nest indices
    where the method evaluates as True.

    Parameters
    ----------
    nest
        The nest to check the leaves of.
    fn
        The conditon function, returning True or False.
    check_nests
        Whether to also check the nests for the condition, not only nest leaves.
        Default is False.
    _index
        The indices detected so far. None at the beginning. Used internally, do not set
        manually.
    _base
        Whether the current function call is the first function call in the recursive
        stack. Used internally, do not set manually.

    Returns
    -------
    ret
        A set of indices for the nest where the function evaluated as True.

    """
    to_ignore = ivy.default(to_ignore, ())
    _index = list() if _index is None else _index
    if isinstance(nest, (tuple, list)) and not isinstance(nest, to_ignore):
        _indices = [
            nested_indices_where(item, fn, check_nests, to_ignore, _index + [i], False)
            for i, item in enumerate(nest)
        ]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    elif isinstance(nest, dict) and not isinstance(nest, to_ignore):
        _indices = [
            nested_indices_where(v, fn, check_nests, to_ignore, _index + [k], False)
            for k, v in nest.items()
        ]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    else:
        cond_met = fn(nest)
        if cond_met:
            return [_index]
        return False
    return [index for index in _indices if index]


def all_nested_indices(
    nest: Iterable, include_nests: bool = False, _index: List = None, _base: bool = True
) -> Union[Iterable, bool]:
    """Checks the leaf nodes of nested x via function fn, and returns all nest indices
    where the method evaluates as True.

    Parameters
    ----------
    nest
        The nest to check the leaves of.
    include_nests
        Whether to also include indices of the nests themselves, not only leaves.
        Default is False.
    _index
        The indices detected so far. None at the beginning. Used internally, do not set
        manually.
    _base
        Whether the current function call is the first function call in the recursive
        stack. Used internally, do not set manually.

    Returns
    -------
    ret
        A set of indices for the nest where the function evaluated as True.

    """
    _index = list() if _index is None else _index
    if isinstance(nest, (tuple, list)):
        _indices = [
            all_nested_indices(item, include_nests, _index + [i], False)
            for i, item in enumerate(nest)
        ]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if include_nests:
            _indices.append(_index)
    elif isinstance(nest, dict):
        _indices = [
            all_nested_indices(v, include_nests, _index + [k], False)
            for k, v in nest.items()
        ]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if include_nests:
            _indices.append(_index)
    else:
        return [_index]
    return [index for index in _indices if index]


# noinspection PyShadowingBuiltins
def map(
    fn: Callable,
    constant: Dict[str, Any] = None,
    unique: Dict[str, Iterable[Any]] = None,
    mean: bool = False,
) -> List:
    """Applies a function on each item of an iterable x.

    Parameters
    ----------
    fn
        The function to map onto x.
    constant
        keyword arguments which remain constant between each function call.
        Default is None.
    unique
        keyword arguments which are unique for each function call. Default is None.
    mean
        Whether to compute the mean across the return values, and return this mean.
        Default is False.

    Returns
    -------
    ret
        x following the applicable of fn to each of it's iterated items.

    """
    c = ivy.default(constant, {})
    u = ivy.default(unique, {})
    rets = [
        r
        for r in _map(
            lambda *uv: fn(**dict(**c, **dict(zip(u.keys(), uv)))), *u.values()
        )
    ]
    if mean:
        return sum(rets) / len(rets)
    return rets


def nested_map(
    x: Union[Union[ivy.Array, ivy.NativeArray], Iterable],
    fn: Callable,
    include_derived: Optional[Union[Dict[type, bool], bool]] = None,
    to_mutable: bool = False,
    max_depth: int = None,
    _depth: int = 0,
    _tuple_check_fn: Optional[callable] = None,
    _list_check_fn: Optional[callable] = None,
    _dict_check_fn: Optional[callable] = None,
) -> Union[Union[ivy.Array, ivy.NativeArray], Iterable, Dict]:
    """Applies a function on x in a nested manner, whereby all dicts, lists and tuples
    are traversed to their lowest leaves before applying the method and returning x. If
    x is not nested, the method is applied to x directly.

    Parameters
    ----------
    x
        The item to apply the mapped function to.
    fn
        The function to map onto x.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is False.
    to_mutable
        Whether to convert the nest to a mutable form, changing all tuples to lists.
        Default is False.
    max_depth
        The maximum nested depth to reach. Default is 1. Increase this if the nest is
        deeper.
    _depth
        Placeholder for tracking the recursive depth, do not set this parameter.
    _tuple_check_fn
        Placeholder for the tuple check function, do not set this parameter.
    _list_check_fn
        Placeholder for the list check function, do not set this parameter.
    _dict_check_fn
        Placeholder for the dict check function, do not set this parameter.

    Returns
    -------
    ret
        x following the applicable of fn to it's nested leaves, or x itself if x is not
        nested.

    """
    if include_derived is True:
        include_derived = {tuple: True, list: True, dict: True}
    elif not include_derived:
        include_derived = {}
    for t in (tuple, list, dict):
        if t not in include_derived:
            include_derived[t] = False
    if ivy.exists(max_depth) and _depth > max_depth:
        return x
    class_instance = type(x)
    tuple_check_fn = ivy.default(
        _tuple_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived[tuple]
        else (lambda x_, t_: type(x_) is t_),
    )
    list_check_fn = ivy.default(
        _list_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived[list]
        else (lambda x_, t_: type(x_) is t_),
    )
    dict_check_fn = ivy.default(
        _dict_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived[dict]
        else (lambda x_, t_: type(x_) is t_),
    )
    if tuple_check_fn(x, tuple):
        ret_list = [
            nested_map(
                i,
                fn,
                include_derived,
                to_mutable,
                max_depth,
                _depth + 1,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
            )
            for i in x
        ]
        if to_mutable:
            return ret_list
        elif hasattr(x, "_fields"):
            # noinspection PyProtectedMember
            return class_instance(**dict(zip(x._fields, ret_list)))
        else:
            return class_instance(ret_list)
    elif list_check_fn(x, list):
        return class_instance(
            [
                nested_map(
                    i,
                    fn,
                    include_derived,
                    to_mutable,
                    max_depth,
                    _depth + 1,
                    tuple_check_fn,
                    list_check_fn,
                    dict_check_fn,
                )
                for i in x
            ]
        )
    elif dict_check_fn(x, dict):
        class_instance = type(x)
        return class_instance(
            {
                k: nested_map(
                    v,
                    fn,
                    include_derived,
                    to_mutable,
                    max_depth,
                    _depth + 1,
                    tuple_check_fn,
                    list_check_fn,
                    dict_check_fn,
                )
                for k, v in x.items()
            }
        )
    return fn(x)


def nested_any(
    nest: Iterable, fn: Callable, check_nests: bool = False, _base: bool = True
) -> bool:
    """Checks the leaf nodes of nest x via function fn, and returns True if any evaluate
    to True, else False.

    Parameters
    ----------
    nest
        The nest to check the leaves of.
    fn
        The conditon function, returning True or False.
    check_nests
        Whether to also check the nests for the condition, not only nest leaves.
        Default is False.
    _base
        Whether the current function call is the first function call in the recursive
        stack. Used internally, do not set manually.

    Returns
    -------
    ret
        A boolean, whether the function evaluates to true for any leaf node.

    """
    if isinstance(nest, (tuple, list)):
        for i, item in enumerate(nest):
            if nested_any(item, fn, check_nests, False):
                return True
        if check_nests and fn(nest):
            return True
    elif isinstance(nest, dict):
        for k, v in nest.items():
            if nested_any(v, fn, check_nests, False):
                return True
        if check_nests and fn(nest):
            return True
    elif fn(nest):
        return True
    return False


def copy_nest(
    nest: Union[Union[ivy.Array, ivy.NativeArray], Iterable],
    include_derived: bool = False,
    to_mutable: bool = False,
) -> Union[ivy.Array, ivy.NativeArray, Iterable, Dict]:
    """Copies a nest deeply, but without copying leaves of the nest, only the nest
    lists, tuples and dicts are copied.

    Parameters
    ----------
    nest
        The nest to copy.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is False.
    to_mutable
        Whether to convert the nest to a mutable form, changing all tuples to lists.
        Default is False.

    Returns
    -------
    ret
        The copied nest.

    """
    class_instance = type(nest)
    check_fn = (
        (lambda x_, t: isinstance(nest, t))
        if include_derived
        else (lambda x_, t: type(nest) is t)
    )
    if check_fn(nest, tuple):
        ret_list = [copy_nest(i, include_derived, to_mutable) for i in nest]
        if to_mutable:
            return ret_list
        return class_instance(tuple(ret_list))
    elif check_fn(nest, list):
        return class_instance([copy_nest(i, include_derived, to_mutable) for i in nest])
    elif check_fn(nest, dict):
        class_instance = type(nest)
        return class_instance(
            {k: copy_nest(v, include_derived, to_mutable) for k, v in nest.items()}
        )
    return nest
