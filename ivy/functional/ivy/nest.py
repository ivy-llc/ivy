"""Collection of Ivy functions for nested objects."""

# global
from builtins import map as _map
from typing import Callable, Any, Union, List, Tuple, Optional, Dict, Iterable

# local
import ivy


# Extra #
# ------#


def index_nest(
    nest: Union[List, Tuple, Dict, ivy.Array, ivy.NativeArray],
    index: Union[List[int], Tuple[int], Iterable[int]],
) -> Any:
    """Index a nested object, using a tuple of indices or keys in the case of dicts.

    Parameters
    ----------
    nest
        The nested object to index.
    index
        A tuple of indices for indexing.

    Returns
    -------
    ret
        The result element through indexing the nested object.

    Examples
    --------
    With :code:`Tuple` inputs:

    >>> x = (1, 2)
    >>> y = [0]
    >>> z = ivy.index_nest(x, y)
    >>> print(z)
    1

    With :code:`ivy.Array` inputs:

    >>> x = ivy.array([[1., 2.], \
                       [3., 4.]])
    >>> y = [1]
    >>> z = ivy.index_nest(x, y)
    >>> print(z)
    ivy.array([3., 4.])

    With :code:`Dict` input:

    >>> x = {'a': 0, 'b': [1, [2, 3]], 'c': (4, 5)}
    >>> y = ('b', 1)
    >>> z = ivy.index_nest(x, y)
    >>> print(z)
    [2, 3]

    With :code:`List` inputs:

    >>> x = [['a', 'b', 'c'], \
             ['d', 'e', 'f'], \
             ['g', ['h', 'i']]]
    >>> y = iter([2, 1, 0])
    >>> z = ivy.index_nest(x, y)
    >>> print(z)
    h
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
        ivy.set_nest_at_index(nest[index[0]], index[1:], value)


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


def set_nest_at_indices(
    nest: Union[List, Tuple, Dict, ivy.Array, ivy.NativeArray],
    indices: Union[List[int], Tuple[int], Iterable[int]],
    values: Union[List[int], Tuple[int], Iterable[int]],
) -> Any:
    """Set the value of a nested item at specified indices with specified values.

    Parameters
    ----------
    nest
        The nested object to update.
    indices
        A tuple of tuples of indices for the indices at which to update.
    values
        The new values for updating.

    Examples
    --------
    With :code:`List` inputs:

    >>> nest = [[1, 2, 3, 4, 5, 6], ['a', 'b', 'c', 'd', 'e', 'f']]
    >>> indices = [[0, 4], [1, 3]]
    >>> values = [111, 'x']
    >>> ivy.set_nest_at_indices(nest, indices, values)
    >>> print(nest)
    [[1, 2, 3, 4, 111, 6], ['a', 'b', 'c', 'x', 'e', 'f']]

    With :code:`Tuple` inputs:

    >>> nest = (['abc', 'xyz', 'pqr'],[1, 4, 'a', 'b'])
    >>> indices = ((0, 1),(1, 2))
    >>> values = ('ivy', 'x')
    >>> ivy.set_nest_at_indices(nest, indices, values)
    >>> print(nest)
    (['abc', 'ivy', 'pqr'], [1, 4, 'x', 'b'])

    With :code:`Dict` input:

    >>> nest = {'a': [1., 2., 3.], 'b': [4., 5., 6.], 'c': [0.]}
    >>> indices = (('a', 1), ('b', 2), ('c', 0))
    >>> values = (11., 22., 33.)
    >>> ivy.set_nest_at_indices(nest, indices, values)
    >>> print(nest)
    {'a': [1.0, 11.0, 3.0], 'b': [4.0, 5.0, 22.0], 'c': [33.0]}

    With :code:`ivy.Array` inputs:

    >>> nest = ivy.array([[1., 2., 3.],[4., 5., 6.]])
    >>> indices = ((0, 1),(1, 2))
    >>> values = (11., 22.)
    >>> ivy.set_nest_at_indices(nest, indices, values)
    >>> print(nest)
    ivy.array([[1., 11., 3.], [4., 5., 22.]])
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


def map_nest_at_indices(nest: Iterable, indices: Tuple, fn: Callable, /):
    """Map a function to the values of a nested item at the specified indices.

    Parameters
    ----------
    nest
        The nested object to update.
    indices
        A tuple of tuples of indices for the indices at which to update.
    fn
        The function to perform on the nest at the given index.
    
    Examples
    --------
    With :code:`List` inputs:

    >>> nest = [['a', 'c', 'e', 'd', 'u', 'k'], ['m', 'n', 'f', 'p', 'q', 't']]
    >>> indices = [[0, 4], [1, 5]]
    >>> fn = lambda x : x + 'b'
    >>> ivy.map_nest_at_indices(nest, indices, fn)
    >>> print(nest)
    [['a', 'c', 'e', 'd', 'ub', 'k'], ['m', 'n', 'f', 'p', 'q', 'tb']]

    With :code:`Tuple` inputs:

    >>> nest = ([-9, 8, -27],[9, -4, -5, 7])
    >>> indices = ((0, 2),(1, 0),(1, 2))
    >>> fn = abs
    >>> ivy.map_nest_at_indices(nest, indices, fn)
    >>> print(nest)
    ([-9, 8, 27], [9, -4, 5, 7])

    With :code:`Dict` input:

    >>> nest = {'a': [8., 16., 22.], 'b': [10., 44., 81.], 'c': [9., 75., 37.]}
    >>> indices = (('a', 2), ('b', 0), ('c', 1))
    >>> fn = lambda x : x + 1
    >>> ivy.map_nest_at_indices(nest, indices, fn)
    >>> print(nest)
    {'a': [8.0, 16.0, 23.0], 'b': [11.0, 44.0, 81.0], 'c': [9.0, 76.0, 37.0]}

    With :code:`ivy.Array` inputs:

    >>> nest = ivy.array([[-9., 8., -17.],[11., -3., 5.]])
    >>> indices = ((0, 1),(1, 1),(1, 2))
    >>> values = lambda x : x ** 2
    >>> ivy.map_nest_at_indices(nest, indices, fn)
    >>> print(nest)
    ivy.array([[-9., 64., -17.], [11., 9., 25.]])
    """
    [map_nest_at_index(nest, index, fn) for index in indices]


def nested_indices_where(
    nest: Iterable,
    fn: Callable,
    check_nests: bool = False,
    to_ignore: Union[type, Tuple[type]] = None,
    _index: List = None,
    _base: bool = True,
    stop_after_n_found: Optional[int] = None,
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
    to_ignore
        Types to ignore when deciding whether to go deeper into the nest or not
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
        n = 0
        _indices = []
        for i, item in enumerate(nest):
            ind = (
                nested_indices_where(
                    item,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [i],
                    False,
                    stop_after_n_found - n,
                )
                if stop_after_n_found is not None
                else nested_indices_where(
                    item, fn, check_nests, to_ignore, _index + [i], False
                )
            )
            if stop_after_n_found is not None and ind:
                if n < stop_after_n_found:
                    n += len(ind)
                    _indices += [ind]
                else:
                    break
            else:
                _indices += [ind]
            if stop_after_n_found is not None and len(_indices) >= stop_after_n_found:
                break
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    elif isinstance(nest, dict) and not isinstance(nest, to_ignore):
        n = 0
        _indices = []
        for k, v in nest.items():
            ind = (
                nested_indices_where(
                    v,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [k],
                    False,
                    stop_after_n_found - n,
                )
                if stop_after_n_found is not None
                else nested_indices_where(
                    v, fn, check_nests, to_ignore, _index + [k], False
                )
            )
            if stop_after_n_found is not None and ind:
                if n < stop_after_n_found:
                    n += len(ind)
                    _indices += [ind]
                else:
                    break
            else:
                _indices += [ind]
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
    x: Union[ivy.Array, ivy.NativeArray, Iterable],
    fn: Callable,
    include_derived: Optional[Union[Dict[type, bool], bool]] = None,
    to_mutable: bool = False,
    max_depth: int = None,
    _depth: int = 0,
    _tuple_check_fn: Optional[callable] = None,
    _list_check_fn: Optional[callable] = None,
    _dict_check_fn: Optional[callable] = None,
) -> Union[ivy.Array, ivy.NativeArray, Iterable, Dict]:
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
    nest: Union[ivy.Array, ivy.NativeArray, Iterable],
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


def nested_multi_map(
    func,
    nests,
    key_chains=None,
    to_apply=True,
    prune_unapplied=False,
    key_chain="",
    config=None,
    to_ivy=True,
):
    """Apply function to all array values from a collection of identically
    structured ivy arrays.

    Parameters
    ----------
    func
        Function to apply to each nest entry.
    nest
        nests to map.
    key_chains
        The key-chains to apply or not apply the method to. Default is None.
    to_apply
        If True, the method will be applied to key_chains, otherwise key_chains will
        be skipped. Default is True.
    prune_unapplied
        Whether to prune key_chains for which the function was not applied,
        otherwise the leftmost nest value is used. Default is False.
    key_chain
        Chain of keys for this dict entry (Default value = '')
    config
        The configuration for the nests. Default is the same as nest0.
    to_ivy
        convert the output to ivy_arrays. Default is True
    Returns
    -------
        nest containing the result of the funciton.

    """
    nest0 = nests[0]
    return_list = list()
    for index, val in enumerate(nest0):
        values = [nest[index] for nest in nests]
        value0 = values[0]
        this_key_chain = (
            str(index) if key_chain == "" else (key_chain + "/" + str(index))
        )
        if (
            (isinstance(value0, ivy.Array) or isinstance(value0, ivy.NativeArray))
            and ivy.get_num_dims(value0) > 0
        ) or (
            isinstance(value0, list)
            or isinstance(value0, tuple)
            or isinstance(value0, dict)
        ):
            ret = ivy.nested_multi_map(
                func,
                values,
                key_chains,
                to_apply,
                prune_unapplied,
                this_key_chain,
                config,
            )
            if ret:
                if ivy.is_array(ret):
                    return_list.insert(index, ivy.to_list(ret))
                else:
                    return_list.insert(index, ret)
        else:
            if key_chains is not None:
                if (this_key_chain in key_chains and not to_apply) or (
                    this_key_chain not in key_chains and to_apply
                ):
                    if prune_unapplied:
                        continue
                    if ivy.is_array(value0):
                        return_list.insert(index, ivy.to_list(value0))
                    else:
                        return_list.insert(index, value0)
                    continue
            ret = func(values, this_key_chain)
            if ivy.is_array(ret):
                return_list.insert(index, ivy.to_list(ret))
            else:
                return_list.insert(index, ret)

    # noinspection PyProtectedMember
    if to_ivy:
        return ivy.array(return_list)
    else:
        return return_list
