from typing import Union, Callable, Any, Iterable
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_like_without_promotion,
    to_native_arrays_and_back,
    to_ivy_arrays_and_back,
)


def if_else(
    cond: bool,
    body_fn: Callable,
    orelse_fn: Callable,
    vars: Iterable[Union[ivy.Array, ivy.NativeArray]],
) -> Any:
    """
    Takes a boolean condition and two functions as input. If the condition is True,
    the first function is executed and its result is returned. Otherwise, the second
    function is executed and its result is returned.

    Parameters
    ----------
    cond
        A boolean value representing the condition to be evaluated.
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
    >>> cond = True
    >>> body_fn = lambda x: x + 1
    >>> orelse_fn = lambda x: x - 1
    >>> vars = (1,)
    >>> result = ivy.if_else(cond, body_fn, orelse_fn, vars)
    >>> print(result)
    2

    >>> cond = False
    >>> body_fn = lambda x: x * 2
    >>> orelse_fn = lambda x: x / 2
    >>> vars = ivy.array([1, 2, 3])
    >>> result = ivy.if_else(cond, body_fn, orelse_fn, vars=(vars,))
    >>> print(result)
    ivy.array([0.5, 1.0, 1.5])

    """

    @to_native_arrays_and_back
    @handle_array_like_without_promotion
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
    Takes a test function, a body function and a set of variables as input. The body
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
    (3,)

    >>> i = 0
    >>> j = 1
    >>> test_fn = lambda i, j: i < 3
    >>> body_fn = lambda i, j: (i + 1, j * 2)
    >>> vars = (i, j)
    >>> result = ivy.while_loop(test_fn, body_fn, vars=vars)
    >>> print(result)
    (3, 8)

    """

    @to_native_arrays_and_back
    @handle_array_like_without_promotion
    def _while_loop(test_fn, body_fn, vars):
        return current_backend().while_loop(test_fn, body_fn, vars)

    test_fn = to_ivy_arrays_and_back(test_fn)
    body_fn = to_ivy_arrays_and_back(body_fn)

    return _while_loop(test_fn, body_fn, vars)
