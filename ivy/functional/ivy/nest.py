"""Collection of Ivy functions for nested objects."""

# global
from builtins import map as _map
from typing import Callable, Any, Union, List, Tuple, Optional, Dict, Iterable, Sequence
import copy
from collections import UserDict, OrderedDict

# local
import ivy
from ivy.utils.exceptions import handle_exceptions


# Extra #
# ------#


@handle_exceptions
def index_nest(
    nest: Union[List, Tuple, Dict, ivy.Array, ivy.NativeArray, ivy.Container],
    index: Union[List[int], Tuple[int], Iterable[int]],
    /,
) -> Any:
    """
    Index a nested object, using a tuple of indices or keys in the case of dicts.

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

    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([[1., 2.],
    ...                [3., 4.]])
    >>> y = [1]
    >>> z = ivy.index_nest(x, y)
    >>> print(z)
    ivy.array([3., 4.])

    With :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a = ivy.array([[1.,2.], [3.,4.]]),
    ...                   b = (50,60))
    >>> y = [1]
    >>> z = ivy.index_nest(x, y)
    >>> print(z)
    >>> z
    {
        a: ivy.array([3., 4.]),
        b: 60
    }

    With :code:`Dict` input:

    >>> x = {'a': 0, 'b': [1, [2, 3]], 'c': (4, 5)}
    >>> y = ('b', 1)
    >>> z = ivy.index_nest(x, y)
    >>> print(z)
    [2, 3]

    With :code:`List` inputs:

    >>> x = [['a', 'b', 'c'],
    ...      ['d', 'e', 'f'],
    ...      ['g', ['h', 'i']]]
    >>> y = iter([2, 1, 0])
    >>> z = ivy.index_nest(x, y)
    >>> print(z)
    h
    """
    ret = nest
    for i in index:
        ret = ret[i]
    return ret


@handle_exceptions
def prune_nest_at_index(nest: Iterable, index: Tuple, /) -> None:
    """
    Prune a nested object at a specified index.

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


@handle_exceptions
def set_nest_at_index(
    nest: Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List, Tuple],
    index: Sequence[Union[str, int]],
    value: Any,
    /,
    shallow: bool = True,
    _result: Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List, Tuple] = None,
) -> Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List, Tuple]:
    """
    Set the value of a nested item at a specified index.

    Parameters
    ----------
    nest
        The nested object to update.
    index
        A tuple of indices for the index at which to update.
    value
        The new value for updating.
    shallow
        Whether to inplace update the input nest or not
        Only works if nest is a mutable type. Default is ``True``.
    _result
        Placeholder for the result of the update. do not set this paramter.

    Returns
    -------
    ret
        nest with changed value at the given index.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([[1., 2.], [3., 4.]])
    >>> y = (1, 1)
    >>> z = 5.
    >>> ivy.set_nest_at_index(x, y, z)
    >>> print(x)
    ivy.array([[1., 2.], [3., 5.]])

    >>> x = ivy.array([1., 2., 3., 4.])
    >>> y = [1]
    >>> z = 5.
    >>> ivy.set_nest_at_index(x, y, z)
    >>> print(x)
    ivy.array([1., 5., 3., 4.])

    With :code:`Dict` input:

    >>> x = {1 : [1, [2, 3]], 2: (4, 5)}
    >>> y = (1, 1)
    >>> z = 2
    >>> ivy.set_nest_at_index(x, y, z)
    >>> print(x)
    {1: [1, 2], 2: (4, 5)}

    With :code:`List` inputs:

    >>> x = [['a', 'b', 'c'],
    ...      ['d', 'e', 'f'],
    ...      ['g', ['h', 'i']]]
    >>> y = (2, 1, 0)
    >>> z = 'H'
    >>> ivy.set_nest_at_index(x, y, z)
    >>> print(x)
    [['a','b','c'],['d','e','f'],['g',['H','i']]]

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1., 2.]) , b=ivy.array([4., 5.]))
    >>> y = ('b',)
    >>> z = ivy.array([3., 4.])
    >>> ivy.set_nest_at_index(x, y, z)
    >>> print(x)
    {
        a: ivy.array([1., 2.]),
        b: ivy.array([3., 4.])
    }
    """
    is_tuple = isinstance(nest, tuple)
    nest_type = type(nest) if is_tuple else lambda x: x
    if _result is None:
        if shallow:
            _result = nest_type(nest)
        else:
            _result = copy_nest(nest, include_derived=True)
    _result = list(_result) if is_tuple else _result
    if len(index) == 1:
        if shallow:
            try:
                nest[index[0]] = value
            except TypeError:
                pass
        _result[index[0]] = value
    else:
        _result[index[0]] = set_nest_at_index(
            nest[index[0]], index[1:], value, shallow, _result[index[0]]
        )
    try:
        _result = nest_type(_result)
    except TypeError:
        _result = nest_type(*_result)
    return _result


@handle_exceptions
def insert_into_nest_at_index(nest: Iterable, index: Tuple, value, /) -> None:
    if len(index) == 1:
        idx = index[0]
        if isinstance(nest, list):
            nest.insert(idx, value)
        else:
            nest[index[0]] = value
    else:
        insert_into_nest_at_index(nest[index[0]], index[1:], value)


@handle_exceptions
def map_nest_at_index(
    nest: Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List],
    index: Sequence[Union[str, int]],
    fn: Callable[[Any], Any],
    /,
    shallow: bool = True,
    _result: Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List] = None,
) -> Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List, Tuple]:
    """
    Map a function to the value of a nested item at a specified index.

    Parameters
    ----------
    nest
        The nested object to update.
    index
        A linear sequence of indices for the index at which to update.
    fn
        The function to perform on the nested value at the given index.
    shallow
        Whether to inplace update the input nest or not
        Only works if nest is a mutable type. Default is ``True``.
    _result
        Placeholder for the result of the update. do not set this paramter.

    Returns
    -------
    ret
        nest with applicable of fn on given index.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([[1., 2.], [3., 4.]])
    >>> y = (1, 1)
    >>> z = lambda a: a + 1.
    >>> ivy.map_nest_at_index(x, y, z)
    >>> print(x)
    ivy.array([[1., 2.], [3., 5.]])

    >>> x = ivy.array([1., 2., 3., 4.])
    >>> y = [1]
    >>> z = lambda a: a + 3.
    >>> ivy.map_nest_at_index(x, y, z)
    >>> print(x)
    ivy.array([1., 5., 3., 4.])

    With :code:`Dict` input:

    >>> x = {1 : [1, [2, 3]], 2: (4, 5)}
    >>> y = (1, 1)
    >>> z = lambda _: 2
    >>> ivy.map_nest_at_index(x, y, z)
    >>> print(x)
    {1: [1, 2], 2: (4, 5)}

    With :code:`List` inputs:

    >>> x = [['a', 'b', 'c'],
    ...      ['d', 'e', 'f'],
    ...      ['g', ['h', 'i']]]
    >>> y = (2, 1, 0)
    >>> z = lambda a: a + 'H'
    >>> ivy.map_nest_at_index(x, y, z)
    >>> print(x)
    [['a','b','c'],['d','e','f'],['g',['hH','i']]]

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1., 2.]) , b=ivy.array([4., 5.]))
    >>> y = ('b',)
    >>> z = lambda _: ivy.array([3., 4.])
    >>> ivy.map_nest_at_index(x, y, z)
    >>> print(x)
    {
        a: ivy.array([1., 2.]),
        b: ivy.array([3., 4.])
    }
    """
    is_tuple = isinstance(nest, tuple)
    nest_type = type(nest) if is_tuple else lambda x: x
    if _result is None:
        if shallow:
            _result = nest_type(nest)
        else:
            _result = copy_nest(nest, include_derived=True)
    _result = list(_result) if is_tuple else _result
    if len(index) == 1:
        ret = fn(nest[index[0]])
        if shallow:
            try:
                nest[index[0]] = ret
            except TypeError:
                pass
        _result[index[0]] = ret
    else:
        _result[index[0]] = map_nest_at_index(
            nest[index[0]], index[1:], fn, shallow, _result[index[0]]
        )
    try:
        _result = nest_type(_result)
    except TypeError:
        _result = nest_type(*_result)
    return _result


@handle_exceptions
def multi_index_nest(
    nest: Union[List, Dict, Tuple, ivy.Array, ivy.NativeArray, ivy.Container],
    indices: Iterable[Iterable[int]],
    /,
) -> Iterable[Any]:
    """
    Repeatedly index a nested object, using a tuple of tuples of indices or keys in the
    case of dicts.

    Parameters
    ----------
    nest
        The nested object to slice.
    indices
        A tuple of tuples of indices to apply.

    Returns
    -------
    ret
        The result elements through indexing the nested object.

    Examples
    --------
    With :code:`Tuple` inputs:

    >>> x = (1, 2)
    >>> y = [[0]]
    >>> z = ivy.multi_index_nest(x, y)
    >>> print(z)
    [1]

    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([[1., 2.],
    ...                [3., 4.]])
    >>> y = [[0],[1]]
    >>> z = ivy.multi_index_nest(x, y)
    >>> print(z)
    [ivy.array([1., 2.], ivy.array([3., 4.])]

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1,2]),
    ...                   b=[30,40])
    >>> y = ('a', ('b', 0))
    >>> z = ivy.multi_index_nest(x, y)
    >>> print(z)
    [ivy.array([1, 2]), 30]

    With :code:`Dict` input:

    >>> x = {'a': 0, 'b': [1, [2, 3]], 'c': (4, 5)}
    >>> y = (('b', 1), 'a')
    >>> z = ivy.multi_index_nest(x, y)
    >>> print(z)
    [[2, 3], 0]

    With :code:`List` inputs:

    >>> x = [['a', 'b', 'c'],
    ...      ['d', 'e', 'f'],
    ...      ['g', ['h', 'i']]]
    >>> y = [[2, 1, 0], [0, 1]]
    >>> z = ivy.multi_index_nest(x, y)
    >>> print(z)
    ['h', 'b']
    """
    return [index_nest(nest, index) for index in indices]


@handle_exceptions
def prune_nest_at_indices(nest: Iterable, indices: Tuple, /) -> None:
    """
    Prune a nested object at specified indices.

    Parameters
    ----------
    nest
        The nested object to prune.
    indices
        A tuple of tuples of indices for the indices at which to prune.
    """
    [prune_nest_at_index(nest, index) for index in indices]


@handle_exceptions
def set_nest_at_indices(
    nest: Union[List, Tuple, Dict, ivy.Array, ivy.NativeArray],
    indices: Union[List[int], Tuple[int], Iterable[int]],
    values: Union[List[int], Tuple[int], Iterable[int]],
    /,
    shallow: bool = True,
) -> Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List, Tuple]:
    """
    Set the value of a nested item at specified indices with specified values.

    Parameters
    ----------
    nest
        The nested object to update.
    indices
        A tuple of tuples of indices for the indices at which to update.
    values
        The new values for updating.
    shallow
        Whether to inplace update the input nest or not
        Only works if nest is a mutable type. Default is ``True``.

    Returns
    -------
    ret
        nest with updated values at the given indices.

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

    >>> nest = [['abc', 'xyz', 'pqr'],[1, 4, 'a', 'b']]
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

    With :class:`ivy.Array` inputs:

    >>> nest = ivy.array([[1., 2., 3.],[4., 5., 6.]])
    >>> indices = ((0, 1),(1, 2))
    >>> values = (11., 22.)
    >>> ivy.set_nest_at_indices(nest, indices, values)
    >>> print(nest)
    ivy.array([[1., 11., 3.], [4., 5., 22.]])
    """
    is_tuple = isinstance(nest, tuple)
    nest_type = type(nest) if is_tuple else lambda x: x
    if shallow:
        result = nest_type(nest)
    else:
        result = copy_nest(nest, include_derived=True)
    result = list(result) if is_tuple else result
    if not isinstance(values, (list, tuple)):
        values = [values] * len(indices)
    for index, value in zip(indices, values):
        result = set_nest_at_index(nest, index, value, _result=result, shallow=shallow)
    try:
        result = nest_type(result)
    except TypeError:
        result = nest_type(*result)
    return result


@handle_exceptions
def insert_into_nest_at_indices(nest: Iterable, indices: Tuple, values, /) -> None:
    """
    Insert a value into the nested item at specified indices with specified values.

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


@handle_exceptions
def map_nest_at_indices(
    nest: Iterable,
    indices: Tuple,
    fn: Callable,
    /,
    shallow: bool = True,
) -> Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List, Tuple]:
    """
    Map a function to the values of a nested item at the specified indices.

    Parameters
    ----------
    nest
        The nested object to update.
    indices
        A tuple of tuples of indices for the indices at which to update.
    fn
        The function to perform on the nest at the given index.
    shallow
        Whether to inplace update the input nest or not
        Only works if nest is a mutable type. Default is ``True``.

    Returns
    -------
    ret
        nest with applicable of fn on given indices.

    Examples
    --------
    With :code:`List` inputs:

    >>> nest = [['a', 'c', 'e', 'd', 'u', 'k'], ['m', 'n', 'f', 'p', 'q', 't']]
    >>> indices = [[0, 4], [1, 5]]
    >>> function = lambda x : x + 'b'
    >>> ivy.map_nest_at_indices(nest, indices, function)
    >>> print(nest)
    [['a', 'c', 'e', 'd', 'ub', 'k'], ['m', 'n', 'f', 'p', 'q', 'tb']]

    With :code:`Tuple` inputs:

    >>> nest = ([-9, 8, -27],[9, -4, -5, 7])
    >>> indices = ((0, 2),(1, 0),(1, 2))
    >>> function = abs
    >>> ivy.map_nest_at_indices(nest, indices, function)
    >>> print(nest)
    ([-9, 8, 27], [9, -4, 5, 7])

    With :code:`Dict` input:

    >>> nest = {'a': [8., 16., 22.], 'b': [10., 44., 81.], 'c': [9., 75., 37.]}
    >>> indices = (('a', 2), ('b', 0), ('c', 1))
    >>> function = lambda x : x + 1
    >>> ivy.map_nest_at_indices(nest, indices, function)
    >>> print(nest)
    {'a': [8.0, 16.0, 23.0], 'b': [11.0, 44.0, 81.0], 'c': [9.0, 76.0, 37.0]}

    With :class:`ivy.Array` inputs:

    >>> nest = ivy.array([[-9., 8., -17.],[11., -3., 5.]])
    >>> indices = ((0, 1),(1, 1),(1, 2))
    >>> function = lambda x : x ** 2
    >>> ivy.map_nest_at_indices(nest, indices, function)
    >>> print(nest)
    ivy.array([[-9., 8., -17.], [11., -3., 5.]])
    """
    is_tuple = isinstance(nest, tuple)
    nest_type = type(nest) if is_tuple else lambda x: x
    if shallow:
        result = nest_type(nest)
    else:
        result = copy_nest(nest, include_derived=True)
    result = list(result) if is_tuple else result
    for i, index in enumerate(indices):
        result = map_nest_at_index(nest, index, fn, _result=result, shallow=shallow)
    try:
        result = nest_type(result)
    except TypeError:
        result = nest_type(*result)
    return result


@handle_exceptions
def nested_argwhere(
    nest: Iterable,
    fn: Callable,
    check_nests: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    _index: Optional[List] = None,
    _base: bool = True,
    stop_after_n_found: Optional[int] = None,
    extra_nest_types: Optional[Union[type, Tuple[type]]] = None,
) -> Union[Iterable, bool]:
    """
    Check the leaf nodes of nested x via function fn, and returns all nest indices where
    the method evaluates as True.

    Parameters
    ----------
    nest
        The nest to check the leaves of.
    fn
        The conditon function, returning True or False.
    check_nests
        Whether to also check the nests for the condition, not only nest leaves.
        Default is ``False``.
    to_ignore
        Types to ignore when deciding whether to go deeper into the nest or not
    _index
        The indices detected so far. None at the beginning. Used internally, do not set
        manually.
    _base
        Whether the current function call is the first function call in the recursive
        stack. Used internally, do not set manually.
    stop_after_n_found
        to stop after some needed indices are found.
    extra_nest_types
        Types to recursively check when deciding whether to go deeper into the
        nest or not

    Returns
    -------
    ret
        A set of indices for the nest where the function evaluated as True.

    Examples
    --------
    With :code:`List` input:

    >>> nest = [[[1, -2, 3], 19], [[9, -36, 80], -10.19]]
    >>> fun = ivy.abs
    >>> nested_indices = ivy.nested_argwhere(nest, fn=fun)
    >>> print(nested_indices)
    [
        [0, 0, 0], [0, 0, 1],
        [0, 0, 2], [0, 1],
        [1, 0, 0], [1, 0, 1],
        [1, 0, 2], [1, 1]
    ]


    With :code:`Tuple` input:

    >>> nest = ([-5, 9, 2], [0.3, 4.])
    >>> fun = ivy.abs
    >>> nested_indices = ivy.nested_argwhere(nest, fn=fun, stop_after_n_found=4)
    >>> print(nested_indices)
    [[0, 0], [0, 1], [0, 2], [1, 0]]

    With :code:`Dict` input:

    >>> nest={'a': [2., 0.6, -2.], 'b': [1., 4., 1.9], 'c': [9.4]}
    >>> fun = ivy.abs
    >>> nested_indices = ivy.nested_argwhere(nest, fn=fun)
    >>> print(nested_indices)
    [
        ['a', 0], ['a', 1],
        ['a', 2], ['b', 0],
        ['b', 1], ['b', 2],
        ['c', 0]
    ]
    """
    to_ignore = ivy.default(to_ignore, ())
    extra_nest_types = ivy.default(extra_nest_types, ())
    _index = list() if _index is None else _index
    if (
        isinstance(nest, (tuple, list)) or isinstance(nest, extra_nest_types)
    ) and not isinstance(nest, to_ignore):
        if isinstance(nest, (ivy.Array, ivy.NativeArray)):
            cond_met = fn(nest)
            ind = ivy.argwhere(cond_met)
            _indices = list()
            for i in range(len(ind)):
                _indices.append(_index + ind.to_list()[i])
            return _indices
        n = 0
        _indices = []
        for i, item in enumerate(nest):
            ind = (
                nested_argwhere(
                    item,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [i],
                    False,
                    stop_after_n_found - n,
                    extra_nest_types,
                )
                if stop_after_n_found is not None
                else nested_argwhere(
                    item,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [i],
                    False,
                    None,
                    extra_nest_types,
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
            if stop_after_n_found is not None and n >= stop_after_n_found:
                break
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    elif (isinstance(nest, dict) or isinstance(nest, UserDict)) and not isinstance(
        nest, to_ignore
    ):
        n = 0
        _indices = []
        for k, v in nest.items():
            ind = (
                nested_argwhere(
                    v,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [k],
                    False,
                    stop_after_n_found - n,
                    extra_nest_types,
                )
                if stop_after_n_found is not None
                else nested_argwhere(
                    v,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [k],
                    False,
                    None,
                    extra_nest_types,
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


@handle_exceptions
def all_nested_indices(
    nest: Union[List, Tuple, Dict, ivy.Array, ivy.NativeArray, ivy.Container] = None,
    /,
    include_nests: bool = False,
    _index: Optional[Union[int, Sequence[int]]] = None,
    _base: bool = True,
    extra_nest_types: Optional[Union[ivy.Dtype, Sequence[ivy.Dtype]]] = None,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return indices of all the elements in nest.

    Parameters
    ----------
    nest
        The nest to check the leaves of.
    include_nests
        Whether to also include indices of the nests themselves, not only
        leaves. Default is ``False``.
    _index
        The indices detected so far. None at the beginning. Used internally,
        do not set manually.
    _base
        Whether the current function call is the first function call in the
        recursive stack. Used internally, do not set manually.
    extra_nest_types
        Types to recursively check when deciding whether to go deeper into the
        nest or not
    out
        Optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        A set of indices of all elements in nest

    Both the description and the type hints above assumes an array input
    for simplicity, but this function is nestable, and therefore also
    accepts :class:ivy.Container instances in place of the arguments.

    Examples
    --------
    With :class:`Dict` input:

    >>> x = {'a': 2., 'b': [6., [15., 9.]], 'c': (7., 56.)}
    >>> y = ivy.all_nested_indices(x)
    >>> print(y)
    [['a'], ['b', 0], ['b', 1, 0], ['b', 1, 1], ['c', 0], ['c', 1]]

    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2., 3., 4.])
    >>> y = ivy.all_nested_indices(x, False, out=x)
    >>> print(y)
    [[]]

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.all_nested_indices(x, True)
    >>> print(y)
    [['a'], ['b']]
    """
    _index = list() if _index is None else _index
    extra_nest_types = ivy.default(extra_nest_types, ())
    if isinstance(nest, (tuple, list)) or isinstance(nest, extra_nest_types):
        if isinstance(nest, (ivy.Array, ivy.NativeArray)):
            ind = ivy.argwhere(ivy.ones_like(nest))
            indices = list()
            for i in range(len(ind)):
                indices.append(_index + ind.to_list()[i])
            return indices
        _indices = [
            all_nested_indices(
                item, include_nests, _index + [i], False, extra_nest_types
            )
            for i, item in enumerate(nest)
        ]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if include_nests:
            _indices.append(_index)
    elif isinstance(nest, dict):
        _indices = [
            all_nested_indices(v, include_nests, _index + [k], False, extra_nest_types)
            for k, v in nest.items()
        ]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if include_nests:
            _indices.append(_index)
    else:
        return [_index]
    return [index for index in _indices if index]


# noinspection PyShadowingBuiltins


@handle_exceptions
def map(
    fn: Callable,
    constant: Optional[Dict[str, Any]] = None,
    unique: Optional[Dict[str, Iterable[Any]]] = None,
    mean: bool = False,
) -> List:
    """
    Apply a function on each item of an iterable x.

    Parameters
    ----------
    fn
        The function to map onto x.
    constant
        keyword arguments which remain constant between each function call.
        Default is ``None``.
    unique
        keyword arguments which are unique for each function call. Default is ``None``.
    mean
        Whether to compute the mean across the return values, and return this mean.
        Default is ``False``.

    Returns
    -------
    ret
        x following the application of fn to each of its iterated items.

    Examples
    --------
    With :code:`int` inputs:

    >>> def special_square(x : float) -> float : return np.square(x)
    >>> results = ivy.map(fn = special_square,
    ...                   constant = None,
    ...                   unique = {'x' : [1,2,3]},
    ...                   mean = False)
    >>> print(results)
    [1, 4, 9]

    >>> results = ivy.map(fn = special_square,
    ...                   constant = None,
    ...                   unique = {'x':[0,1,2]},
    ...                   mean = True)
    >>> print(results)
    1.6666666666666667

    >>> def special_pow(x:float,y:float) ->float : return np.power(x,y)
    >>> results = ivy.map(fn = special_pow,
    ...                   constant = {'y':[0,1]},
    ...                   unique = {'x':[1,2,3]},
    ...                   mean = False)
    >>> print(results)
    [array([1,1]),
    array([1,2]),
    array([1,3])]

    >>> results = ivy.map(fn = special_pow,
    ...                   constant = {'y':[0,1]},
    ...                   unique = {'x':[1,2,3]},
    ...                   mean = True)
    >>> print(results)
    [1. 2.]

    With float inputs:

    >>> def linear_model(w:float, x:float, b:float) -> float: return w*x + b
    >>> results = ivy.map(fn = linear_model,
    ...                   constant = {'w':10., 'b':1.},
    ...                   unique = {'x':[0.,1.,2.]},
    ...                   mean = False)
    >>> print(results)
    [1.0, 11.0, 21.0]

    With :class:`ivy.Array` inputs:

    >>> results = ivy.map(fn = linear_model,
    ...    constant = {'w':ivy.array([1.,0.,1.]), 'b':ivy.array([0.,10.,100.])},
    ...    unique = {'x':[ivy.array([0.,1.,0.]), ivy.array([1.,1.,1.])]},
    ...    mean = False)
    >>> print(results)
    [ivy.array([0., 10., 100.]),
    ivy.array([1., 10., 101.])]

    >>> results = ivy.map(fn = linear_model,
    ...    constant = {'w':ivy.array([1.,0.,1.]), 'b':ivy.array([0.,10.,100.])},
    ...    unique = {'x':[ivy.array([0.,1.,0.]), ivy.array([1.,1.,1.])]},
    ...    mean = True)
    >>> print(results)
    ivy.array([  0.5,  10. , 100. ])
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
        rets = sum(rets) / len(rets)

    return rets


@handle_exceptions
def nested_map(
    x: Union[ivy.Array, ivy.NativeArray, Iterable],
    /,
    fn: Callable,
    include_derived: Optional[Union[Dict[type, bool], bool]] = None,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    to_mutable: bool = False,
    max_depth: Optional[int] = None,
    _depth: int = 0,
    _tuple_check_fn: Optional[Callable] = None,
    _list_check_fn: Optional[Callable] = None,
    _dict_check_fn: Optional[Callable] = None,
    extra_nest_types: Optional[Union[type, Tuple[type]]] = None,
    shallow: bool = True,
) -> Union[ivy.Array, ivy.NativeArray, Iterable, Dict]:
    """
    Apply a function on x in a nested manner, whereby all dicts, lists and tuples are
    traversed to their lowest leaves before applying the method and returning x. If x is
    not nested, the method is applied to x directly.

    Parameters
    ----------
    x
        The item to apply the mapped function to.
    fn
        The function to map onto x.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    to_ignore
        Types to ignore when deciding whether to go deeper into the nest or not
    to_mutable
        Whether to convert the nest to a mutable form, changing all tuples to lists.
        Default is ``False``.
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
    extra_nest_types
        Types to recursively check when deciding whether to go deeper into the
        nest or not
    shallow
        Whether to inplace update the input nest or not
        Only works if nest is a mutable type. Default is ``True``.

    Returns
    -------
    ret
        x following the applicable of fn to it's nested leaves, or x itself if x is not
        nested.

    Examples
    --------
    With :class:`Tuple` inputs:

    >>> x = ([[1., 2.], [3., 4.]])
    >>> function = lambda a : a * 2
    >>> ivy.nested_map(x, function)
    >>> print(x)
    ([[2.0, 4.0], [6.0, 8.0]])

    With :code:`Dict` input:

    >>> x = {1 : [1, [2, 3]], 2: (4, 5)}
    >>> function = lambda a : a + 1
    >>> ivy.nested_map(x, function)
    >>> print(x)
    {1 : [2, [3, 4]], 2: (5, 6)}

    With :code:`List` inputs:

    >>> x = [['a', 'b', 'c'],
    ...      ['d', 'e', 'f'],
    ...      ['g', ['h', 'i']]]
    >>> function = lambda a: a + 'H'
    >>> ivy.nested_map(x, function)
    >>> print(x)
    [['aH','bH','cH'],['dH','eH','fH'],['gH',['hH','iH']]]

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(
    ...   a=ivy.array([[1, 2, 3], [9, 8, 7]]) , b=ivy.array([[4, 5, 6], [12, 13, 14]])
    ... )
    >>> function = lambda a : a  + 1
    >>> ivy.nested_map(x, function)
    >>> print(x)
    {
       a: ivy.array([[1, 2, 3], [9, 8, 7]]),
       b: ivy.array([[4, 5,  6], [12, 13, 14]])
    }

    >>> nest = [[1, 2], [3, 4], [5, 6], {"a": 1, "b": 2, "c": 3}]
    >>> ivy.nested_map(lambda x: x * 2, nest, to_ignore=(list))
    [[1, 2], [3, 4], [5, 6], {"a": 2, "b": 4, "c": 6}]

    >>> nest = [[1, 2], [3, [4, 5]], [[6], [7, 8, [9, 10]]]]
    >>> ivy.nested_map(lambda x: x * 2, nest, max_depth=1)
    [[2, 4], [6, [4, 5]], [[6], [14, 16, [9, 10]]]]

    >>> nest = ([23, 25, 1337], [63, 98, 6])
    >>> function = lambda a :  a + 1
    >>> ivy.nested_map(nest, function, to_mutable = True)
    >>> print(nest)
    [[24, 25, 1338], [64, 99, 7]]
    """
    to_ignore = ivy.default(to_ignore, ())
    extra_nest_types = ivy.default(extra_nest_types, ())
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
        (
            (lambda x_, t_: isinstance(x_, t_))
            if include_derived[tuple]
            else (lambda x_, t_: type(x_) is t_)
        ),
    )
    list_check_fn = ivy.default(
        _list_check_fn,
        (
            (lambda x_, t_: isinstance(x_, t_))
            if include_derived[list]
            else (lambda x_, t_: type(x_) is t_)
        ),
    )
    dict_check_fn = ivy.default(
        _dict_check_fn,
        (
            (lambda x_, t_: isinstance(x_, t_))
            if include_derived[dict]
            else (lambda x_, t_: type(x_) is t_)
        ),
    )

    if tuple_check_fn(x, tuple) and not isinstance(x, to_ignore):
        ret_list = [
            nested_map(
                i,
                fn,
                include_derived,
                to_ignore,
                to_mutable,
                max_depth,
                _depth + 1,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                extra_nest_types,
                shallow,
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
    elif (list_check_fn(x, list) or isinstance(x, extra_nest_types)) and not isinstance(
        x, to_ignore
    ):
        if isinstance(x, (ivy.Array, ivy.NativeArray)):
            ret = fn(x)
            if shallow:
                return ivy.inplace_update(x, ret)
            return ret
        ret_list = [
            nested_map(
                i,
                fn,
                include_derived,
                to_ignore,
                to_mutable,
                max_depth,
                _depth + 1,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                extra_nest_types,
                shallow,
            )
            for i in x
        ]
        if shallow:
            x[:] = ret_list[:]
            return x
        return class_instance(ret_list)
    elif (dict_check_fn(x, dict) or isinstance(x, UserDict)) and not isinstance(
        x, to_ignore
    ):
        class_instance = type(x)
        ret = {
            k: nested_map(
                v,
                fn,
                include_derived,
                to_ignore,
                to_mutable,
                max_depth,
                _depth + 1,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                extra_nest_types,
                shallow,
            )
            for k, v in x.items()
        }
        if shallow:
            x.update(ret)
            return x
        return class_instance(ret)
    elif isinstance(x, slice):
        # TODO: add tests for this
        return slice(*nested_map([x.start, x.stop, x.step], fn))
    return fn(x)


@handle_exceptions
def nested_any(
    nest: Iterable,
    fn: Callable,
    check_nests: bool = False,
    _base: bool = True,
    extra_nest_types: Optional[Union[type, Tuple[type]]] = None,
) -> bool:
    """
    Check the leaf nodes of nest x via function fn, and returns True if any evaluate to
    True, else False.

    Parameters
    ----------
    nest
        The nest to check the leaves of.
    fn
        The conditon function, returning True or False.
    check_nests
        Whether to also check the nests for the condition, not only nest leaves.
        Default is ``False``.
    _base
        Whether the current function call is the first function call in the recursive
        stack. Used internally, do not set manually.
    extra_nest_types
        Types to recursively check when deciding whether to go deeper into the
        nest or not

    Returns
    -------
    ret
        A boolean, whether the function evaluates to true for any leaf node.
    """
    extra_nest_types = ivy.default(extra_nest_types, ())
    if isinstance(nest, (tuple, list)) or isinstance(nest, extra_nest_types):
        if isinstance(nest, (ivy.Array, ivy.NativeArray)):
            if ivy.any(fn(nest)):
                return True
        for i, item in enumerate(nest):
            if nested_any(item, fn, check_nests, False, extra_nest_types):
                return True
        if check_nests and fn(nest):
            return True
    elif isinstance(nest, dict):
        for k, v in nest.items():
            if nested_any(v, fn, check_nests, False, extra_nest_types):
                return True
        if check_nests and fn(nest):
            return True
    elif fn(nest):
        return True
    return False


@handle_exceptions
def copy_nest(
    nest: Union[ivy.Array, ivy.NativeArray, Iterable],
    /,
    include_derived: bool = False,
    to_mutable: bool = False,
    extra_nest_types: Optional[Union[type, Tuple[type]]] = None,
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
    """
    Copy a nest deeply, but without copying leaves of the nest, only the nest lists,
    tuples and dicts are copied.

    Parameters
    ----------
    nest
        The nest to copy.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    to_mutable
        Whether to convert the nest to a mutable form, changing all tuples to lists.
        Default is ``False``.
    extra_nest_types
        Types to recursively check when deciding whether to go deeper into the
        nest or not

    Returns
    -------
    ret
        The copied nest.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> nest = ivy.array([[1.,2.,3.],[7.,8.,9.]])
    >>> copied_nest = ivy.copy_nest(nest)
    >>> print(copied_nest)
    ivy.array([[1., 2., 3.],
            [7., 8., 9.]])

    With :code:`Iterable` input:

    >>> nest = [[1, 2, 3, 4, 5], [23, 24, 25, 26, 27]]
    >>> copied_nest = ivy.copy_nest(nest, include_derived = True)
    >>> print(copied_nest)
    [[1, 2, 3, 4, 5], [23, 24, 25, 26, 27]]

    >>> nest = ([23, 25, 1337], [63, 98, 6])
    >>> copied_nest = ivy.copy_nest(nest, to_mutable = True)
    >>> print(copied_nest)
    [[23, 25, 1337], [63, 98, 6]]

    >>> nest = {'first': [23., 24., 25], 'second': [46., 48., 50]}
    >>> copied_nest = ivy.copy_nest(nest)
    >>> print(copied_nest)
    {'first': [23.0, 24.0, 25], 'second': [46.0, 48.0, 50]}
    """
    extra_nest_types = ivy.default(extra_nest_types, ())
    class_instance = type(nest)
    check_fn = (
        (lambda x_, t: isinstance(nest, t))
        if include_derived
        else (lambda x_, t: type(nest) is t)
    )
    if check_fn(nest, tuple):
        ret_list = [
            copy_nest(
                i,
                include_derived=include_derived,
                to_mutable=to_mutable,
                extra_nest_types=extra_nest_types,
            )
            for i in nest
        ]
        if to_mutable:
            return ret_list
        if hasattr(nest, "_fields"):
            return class_instance(**dict(zip(nest._fields, ret_list)))
        return class_instance(tuple(ret_list))
    elif check_fn(nest, list) or isinstance(nest, extra_nest_types):
        if isinstance(nest, (ivy.Array, ivy.NativeArray)):
            return copy.deepcopy(nest)
        return class_instance(
            [
                copy_nest(
                    i,
                    include_derived=include_derived,
                    to_mutable=to_mutable,
                    extra_nest_types=extra_nest_types,
                )
                for i in nest
            ]
        )
    elif check_fn(nest, dict):
        class_instance = type(nest)
        dict_ = {
            k: copy_nest(
                v,
                include_derived=include_derived,
                to_mutable=to_mutable,
                extra_nest_types=extra_nest_types,
            )
            for k, v in nest.items()
        }
        if isinstance(nest, OrderedDict):
            return class_instance(**dict_)
        return class_instance(dict_)
    return nest


@handle_exceptions
def nested_multi_map(
    func: Callable,
    nests: List[Iterable],
    index_chains=None,
    to_apply=True,
    prune_unapplied=False,
    index_chain="",
    config=None,
    to_ivy=True,
):
    """
    Apply function to all array values from a collection of identically structured ivy
    arrays.

    Parameters
    ----------
    func
        Function to apply to each nest entry.
    nest
        nests to map.
    index_chains
        The key-chains to apply or not apply the method to. Default is ``None``.
    to_apply
        If True, the method will be applied to index_chains, otherwise index_chains will
        be skipped. Default is ``True``.
    prune_unapplied
        Whether to prune index_chains for which the function was not applied,
        otherwise the leftmost nest value is used. Default is ``False``.
    index_chain
        Chain of keys for this dict entry (Default value = '')
    config
        The configuration for the nests. Default is the same as nest0.
    to_ivy
        convert the output to ivy_arrays. Default is ``True``
    Returns
    -------
        nest containing the result of the function. The structure of the output is the
        same as the input with the result of the function applied to each applicable
        leaf and the value at that leaf in the first nest for a non-applicable leaf if
        prune_unapplied is False else unapplied leaves are pruned.
    """
    nest0 = None
    for nest in nests:
        if isinstance(nest, (tuple, list, dict)):
            nest0 = nest
            break
    if isinstance(nest0, (list, tuple)):
        return_nest = []
    elif isinstance(nest0, dict):
        return_nest = {}
    else:
        return_nest = None
    if nest0 is not None:
        is_dict = isinstance(nest0, dict)
        for index, val in enumerate(nest0):
            if is_dict:
                values = [
                    (
                        nest[index]
                        if isinstance(nest, (tuple, list))
                        else nest[val] if isinstance(nest, dict) else nest
                    )
                    for nest in nests
                ]
            else:
                values = [
                    (
                        nest[index]
                        if isinstance(nest, (tuple, list))
                        else nest[list(nest)[index]] if isinstance(nest, dict) else nest
                    )
                    for nest in nests
                ]
            value0 = values[0]
            if is_dict:
                key = str(index) if isinstance(nest, (tuple, list)) else val
            else:
                key = (
                    str(index) if isinstance(nest, (tuple, list)) else list(nest)[index]
                )
            this_index_chain = key if index_chain == "" else (index_chain + "/" + key)
            ret = ivy.nested_multi_map(
                func,
                values,
                index_chains,
                to_apply,
                prune_unapplied,
                this_index_chain,
                config,
                to_ivy,
            )
            if ret is not None:
                if to_ivy and isinstance(nest, (ivy.Array, ivy.NativeArray)):
                    ret = ivy.array(ivy.to_list(ret))
                (
                    return_nest.append(ret)
                    if isinstance(return_nest, (list))
                    else return_nest.update(
                        {val if is_dict else list(nest)[index]: ret}
                    )
                )
    else:
        values = nests
        value0 = values[0]
        this_index_chain = index_chain

        def _found_in_index_chains(this_index_chain, index_chains):
            if index_chains is None:
                return False
            for index_chain in index_chains:
                if this_index_chain.startswith(index_chain):
                    return True
            return False

        if index_chains is not None:
            found = _found_in_index_chains(this_index_chain, index_chains)
            if (found and not to_apply) or (not found and to_apply):
                if prune_unapplied:
                    return return_nest
                if ivy.is_array(value0):
                    if not to_ivy:
                        value0 = ivy.array(value0)
                (
                    return_nest.append(value0)
                    if isinstance(return_nest, list)
                    else (
                        return_nest.update({this_index_chain: value0})
                        if isinstance(return_nest, dict)
                        else return_nest
                    )
                )
                return (
                    tuple(return_nest)
                    if isinstance(nest, tuple)
                    else (
                        ivy.Container(return_nest)
                        if ivy.is_ivy_container(nest)
                        else return_nest
                    )
                )
        ret = func(values, this_index_chain)
        if to_ivy:
            if isinstance(nest, (ivy.Array, ivy.NativeArray)):
                return ret
            else:
                return ivy.array(ret)
        else:
            return ret
    if prune_unapplied and len(return_nest) == 0:
        return None
    return (
        tuple(return_nest)
        if isinstance(nest0, tuple)
        else ivy.Container(return_nest) if ivy.is_ivy_container(nest0) else return_nest
    )


@handle_exceptions
def duplicate_array_index_chains(nest: Union[ivy.Array, ivy.NativeArray, Iterable]):
    """
    Group all unique index chains in a nest. This function is useful for finding all
    unique index chains in a nest, and then duplicating the values at those index chains
    for functional frameworks.

    Parameters
    ----------
    nest
        nest to get duplicate index chains for.

    Returns
    -------
        list of index chains to duplicate.
    """
    all_index_chains = ivy.nested_argwhere(nest, lambda _: True)
    duplicates = []
    duplicate_index_chains = {}
    for index_chain in all_index_chains:
        val = ivy.index_nest(nest, index_chain)
        if ivy.is_array(val):
            for i in range(len(duplicates)):
                if val is duplicates[i]:
                    duplicate_index_chains[i].append(index_chain)
                    break
            else:
                duplicates.append(val)
                duplicate_index_chains[len(duplicates) - 1] = [index_chain]
    return list(duplicate_index_chains.values())


def prune_empty(nest):
    """
    Prune empty nests from a nest.

    Parameters
    ----------
    nest
        nest to prune.

    Returns
    -------
        pruned nest with all empty nests removed
    """
    valid = False
    if isinstance(nest, dict):
        keys = [k for k in nest]
        for k in keys:
            nest[k] = prune_empty(nest[k])
            if nest[k] is not None:
                valid = True
        for k in keys:
            if nest[k] is None:
                del nest[k]
    elif isinstance(nest, (list, tuple)):
        nest = list(nest)
        for i in range(len(nest)):
            nest[i] = prune_empty(nest[i])
            if nest[i] is not None:
                valid = True
        for i in range(len(nest) - 1, -1, -1):
            if nest[i] is None:
                del nest[i]
    if not valid and not (ivy.is_array(nest) or isinstance(nest, (int, float, str))):
        return None
    return nest
