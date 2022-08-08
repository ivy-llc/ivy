"""Collection of Ivy functions for nested objects."""

# global
from builtins import map as _map
from typing import Callable, Any, Union, List, Tuple, Optional, Dict, Iterable, Sequence

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


def prune_nest_at_index(nest: Iterable, index: Tuple):
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


def set_nest_at_index(
    nest: Union[ivy.Array, ivy.NativeArray, ivy.Container, Dict, List],
    index: Sequence[Union[str, int]],
    value: Any,
):
    """Set the value of a nested item at a specified index.

    Parameters
    ----------
    nest
        The nested object to update.
    index
        A tuple of indices for the index at which to update.
    value
        The new value for updating.

    Examples
    --------
    With :code:`ivy.Array` inputs:
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
    >>> x = [['a', 'b', 'c'], \
             ['d', 'e', 'f'], \
             ['g', ['h', 'i']]]
    >>> y = (2, 1, 0)
    >>> z = 'H'
    >>> ivy.set_nest_at_index(x, y, z)
    >>> print(x)
    [['a','b','c'],['d','e','f'],['g',['H','i']]]

     With :code:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([1., 2.]) , b=ivy.array([4., 5.]))
    >>> y = ('b',)
    >>> z = ivy.array([3., 4.])
    >>> ivy.set_nest_at_index(x, y, z)
    >>> print(x)
    {\
    a: ivy.array([1., 2.]),\
    b: ivy.array([3., 4.])\
    }\
    """
    if len(index) == 1:
        nest[index[0]] = value
    else:
        ivy.set_nest_at_index(nest[index[0]], index[1:], value)


def insert_into_nest_at_index(nest: Iterable, index: Tuple, value):
    if len(index) == 1:
        idx = index[0]
        if isinstance(nest, list):
            nest.insert(idx, value)
        else:
            nest[index[0]] = value
    else:
        insert_into_nest_at_index(nest[index[0]], index[1:], value)


def map_nest_at_index(nest: Iterable, index: Tuple, fn: Callable):
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


def multi_index_nest(nest: Iterable, indices: Tuple):
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


def prune_nest_at_indices(nest: Iterable, indices: Tuple):
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


def insert_into_nest_at_indices(nest: Iterable, indices: Tuple, values):
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


def map_nest_at_indices(nest: Iterable, indices: Tuple, fn: Callable):
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
    stop_after_n_found
        to stop after some needed indices are found.

    Returns
    -------
    ret
        A set of indices for the nest where the function evaluated as True.

    Examples
    --------
    With :code:`List` input:

    >>> nest = [[[1, -2, 3], 19], [[9, -36, 80], -10.19]]
    >>> fun = ivy.abs
    >>> nested_indices = ivy.nested_indices_where(nest, fn=fun)
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
    >>> nested_indices = ivy.nested_indices_where(nest, fn=fun, stop_after_n_found=4)
    >>> print(nested_indices)
    [[0, 0], [0, 1], [0, 2], [1, 0]]

    With :code:`Dict` input:

    >>> nest={'a': [2., 0.6, -2.], 'b': [1., 4., 1.9], 'c': [9.4]}
    >>> fun = ivy.abs
    >>> nested_indices = ivy.nested_indices_where(nest, fn=fun)
    >>> print(nested_indices)
    [
        ['a', 0], ['a', 1],
        ['a', 2], ['b', 0],
        ['b', 1], ['b', 2],
        ['c', 0]
    ]
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
        x following the application of fn to each of its iterated items.
    
    Examples
    --------
    With :code:`int` inputs:

    >>> def special_square(x : float) -> float : return np.square(x)
    >>> results = ivy.map(fn = special_square, \
                          constant = None, \
                          unique = {'x' : [1,2,3]}, \
                          mean = False)
    >>> print(results)
    [1, 4, 9]

    >>> results = ivy.map(fn = special_square, \
                          constant = None, \
                          unique = {'x':[0,1,2]},\
                          mean = True)
    >>> print(results)
    1.6666666666666667

    >>> def special_pow(x:float,y:float) ->float : return np.power(x,y)
    >>> results = ivy.map(fn = special_pow, \
                          constant = {'y':[0,1]}, \
                          unique = {'x':[1,2,3]}, \
                          mean = False)
    >>> print(results)
    [array([1,1]),
    array([1,2]),
    array([1,3])]

    >>> results = ivy.map(fn = special_pow, \
                          constant = {'y':[0,1]}, \
                          unique = {'x':[1,2,3]}, \
                          mean = True)
    >>> print(results)
    [1. 2.]

    With :code:`float` inputs:

    >>> def linear_model(w:float, x:float, b:float) -> float: return w*x + b
    >>> results = ivy.map(fn = linear_model, \
                          constant = {'w':10., 'b':1.}, \
                          unique = {'x':[0.,1.,2.]}, \
                          mean = False)
    >>> print(results)
    [1.0, 11.0, 21.0]

    With :code:`ivy.Array` inputs:

    >>> results = ivy.map(fn = linear_model, \
        constant = {'w':ivy.array([1.,0.,1.]), 'b':ivy.array([0.,10.,100.])}, \
        unique = {'x':[ivy.array([0.,1.,0.]), ivy.array([1.,1.,1.])]}, \
        mean = False)
    >>> print(results)
    [ivy.array([0., 10., 100.]),
    ivy.array([1., 10., 101.])]

    >>> results = ivy.map(fn = linear_model, \
        constant = {'w':ivy.array([1.,0.,1.]), 'b':ivy.array([0.,10.,100.])}, \
        unique = {'x':[ivy.array([0.,1.,0.]), ivy.array([1.,1.,1.])]}, \
        mean = True)
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
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
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

    Examples
    --------
    With :code:`ivy.Array` input:

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
    func: Callable,
    nests: List[Iterable],
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
    nest0 = None
    for nest in nests:
        if isinstance(nest, (tuple, list, dict)):
            nest0 = nest
            break
    return_list = list()
    if nest0 is not None:
        is_dict = isinstance(nest0, dict)
        for index, val in enumerate(nest0):
            if is_dict:
                values = [
                    nest[index]
                    if isinstance(nest, (tuple, list))
                    else nest[val]
                    if isinstance(nest, dict)
                    else nest
                    for nest in nests
                ]
            else:
                values = [
                    nest[index]
                    if isinstance(nest, (tuple, list))
                    else nest[list(nest)[index]]
                    if isinstance(nest, dict)
                    else nest
                    for nest in nests
                ]
            value0 = values[0]
            this_key_chain = (
                str(index) if key_chain == "" else (key_chain + "/" + str(index))
            )
            ret = ivy.nested_multi_map(
                func,
                values,
                key_chains,
                to_apply,
                prune_unapplied,
                this_key_chain,
                config,
                to_ivy,
            )
            if to_ivy:
                if isinstance(nest, (ivy.Array, ivy.NativeArray)):
                    return_list.insert(index, ivy.array(ivy.to_list(ret)))
                else:
                    return_list.insert(index, ret)
            else:
                return_list.insert(index, ret)
    else:
        values = nests
        this_key_chain = key_chain
        if key_chains is not None:
            if (this_key_chain in key_chains and not to_apply) or (
                this_key_chain not in key_chains and to_apply
            ):
                if prune_unapplied:
                    return return_list
                if ivy.is_array(value0):
                    if to_ivy:
                        return_list.append(value0)
                    else:
                        return_list.append(ivy.array(value0))
                else:
                    return_list.append(value0)
                return return_list
        ret = func(values, this_key_chain)
        if to_ivy:
            if isinstance(nest, (ivy.Array, ivy.NativeArray)):
                return ret
            else:
                return ivy.array(ret)
        else:
            return ret
    return return_list
