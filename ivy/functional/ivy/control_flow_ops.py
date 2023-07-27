from typing import Union, Callable, Any, Iterable
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_like_without_promotion,
    to_native_arrays_and_back,
    to_ivy_arrays_and_back,
    handle_device_shifting,
)


def if_else(
    cond: Callable,
    body_fn: Callable,
    orelse_fn: Callable,
    vars: Iterable[Union[ivy.Array, ivy.NativeArray]],
) -> Any:
    """
    Take a condition function and two functions as input. If the condition is True, the
    first function is executed and its result is returned. Otherwise, the second
    function is executed and its result is returned.

    Parameters
    ----------
    cond
        A function returning a boolean.
    body_fn
        A callable function to be executed if the condition is True.
    orelse_fn
        A callable function to be executed if the condition is False.
    vars
        Additional variables to be passed to the functions.

    Returns
    -------
    ret
        The result of executing either body_fn or orelse_fn depending on the value of
        cond.

    Examples
    --------
    >>> x = 1
    >>> cond = x > 0
    >>> body_fn = lambda x: x + 1
    >>> orelse_fn = lambda x: x - 1
    >>> vars = (1,)
    >>> result = ivy.if_else(cond, body_fn, orelse_fn, vars)
    >>> print(result)
    2

    >>> x = 0
    >>> cond = x - 2 == 0
    >>> body_fn = lambda x: x * 2
    >>> orelse_fn = lambda x: x / 2
    >>> vars = ivy.array([1, 2, 3])
    >>> result = ivy.if_else(cond, body_fn, orelse_fn, vars=(vars,))
    >>> print(result)
    ivy.array([0.5, 1. , 1.5])
    """

    @to_native_arrays_and_back
    @handle_array_like_without_promotion
    @handle_device_shifting
    def _if_else(cond, body_fn, orelse_fn, vars):
        return current_backend().if_else(cond, body_fn, orelse_fn, vars)

    body_fn = to_ivy_arrays_and_back(body_fn)
    orelse_fn = to_ivy_arrays_and_back(orelse_fn)

    return _if_else(cond, body_fn, orelse_fn, vars)


def while_loop(
    test_fn: Callable,
    body_fn: Callable,
    vars: Iterable[Union[ivy.Array, ivy.NativeArray]],
) -> Any:
    """
    Take a test function, a body function and a set of variables as input. The body
    function is executed repeatedly while the test function returns True.

    Parameters
    ----------
    test_fn
        A callable function that returns a boolean value representing whether the
        loop should continue.
    body_fn
        A callable function to be executed repeatedly while the test function returns
        True.
    vars
        Additional variables to be passed to the functions.

    Returns
    -------
    ret
        The final result of executing the body function.

    Examples
    --------
    >>> i = 0
    >>> test_fn = lambda i: i < 3
    >>> body_fn = lambda i: i + 1
    >>> result = ivy.while_loop(test_fn, body_fn, vars= (i,))
    >>> print(result)
    (11,)

    >>> i = 0
    >>> j = 1
    >>> test_fn = lambda i, j: i < 3
    >>> body_fn = lambda i, j: (i + 1, j * 2)
    >>> vars = (i, j)
    >>> result = ivy.while_loop(test_fn, body_fn, vars=vars)
    >>> print(result)
    (11,1)
    """

    @to_native_arrays_and_back
    @handle_array_like_without_promotion
    @handle_device_shifting
    def _while_loop(test_fn, body_fn, vars):
        return current_backend().while_loop(test_fn, body_fn, vars)

    test_fn = to_ivy_arrays_and_back(test_fn)
    body_fn = to_ivy_arrays_and_back(body_fn)

    return _while_loop(test_fn, body_fn, vars)


def for_loop(
    iterable: Iterable[Any],
    body_fn: Callable,
    vars: Iterable[Union[ivy.Array, ivy.NativeArray]],
):
    """
    Loops over an iterable, passing the current iteration along with a tuple of
    variables into the provided body function.

    Parameters
    ----------
    iterable
        The iterable to loop over.
    body_fn
        A function to call each iteration, first taking the iterator value
        and then a tuple of extra parameters.
    vars
        Extra parameters to be passed to body_fn.

    Returns
    -------
    ret
        The loop's return value (if any).

    Example
    ----
    ```
    def body_fn(k, args):
        print(k+1)
        return args

    lst = [5,6]

    ivy.for_loop(lst, body_fn, ())
    >>> 5
    >>> 6
    ```
    """
    iterator = iterable.__iter__()

    vars_dict = _tuple_to_dict(vars)

    def test_fn(iterator, original_body, vars_dict):
        try:
            val = iterator.__next__()
        except StopIteration:
            return False

        vars_tuple = original_body(val, _dict_to_tuple(vars_dict))

        for k in range(len(vars_tuple)):
            vars_dict[k] = vars_tuple[k]

        return True

    def empty_function(iterator, original_body, vars_dict):
        return (iterator, original_body, vars_dict)

    packed_vars = (iterator, body_fn, vars_dict)

    return _dict_to_tuple(while_loop(test_fn, empty_function, packed_vars)[2])


# todo (nightcrab) find a better place for these cmp functions


def cmp_is(left, right):
    return left is right


def cmp_isnot(left, right):
    return left is not right


def _tuple_to_dict(t):
    return {k: t[k] for k in range(len(t))}


def _dict_to_tuple(d):
    return tuple([d[k] for k in d])
