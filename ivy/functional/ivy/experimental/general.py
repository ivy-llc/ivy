# global
import functools
from typing import Callable, Union

# local
import ivy
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
def reduce(
    operand: Union[ivy.Array, ivy.NativeArray],
    init_value: Union[int, float, -float("inf"), float("inf")],
    func: Callable,
    axis: int,
) -> ivy.Array:
    """
    Reduces the input array's dimensions by one by applying a function along one axis.

    Parameters
    ----------
    operand
        The array to act on.
    init_value
        The value with which to start the reduction.
    func
        The reduction function.
    axis
        The dimension along which the reduction is performed.

    Returns
    -------
    ret
        The reduced array.

    Examples
    --------
    >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
    >>> ivy.reduce(x, 0, ivy.add, 0)
    ivy.array([6, 15])

    >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
    >>> ivy.reduce(x, 0, ivy.add, 1)
    ivy.array([5, 7, 9])
    """
    func = ivy.output_to_native_arrays(func)
    op_parts = ivy.moveaxis(operand, axis, 0).reshape((operand.shape[axis], -1))
    result = functools.reduce(func, op_parts, init_value)
    result = ivy.reshape(result, operand.shape[:axis] + operand.shape[axis+1:])
    return ivy.inputs_to_native_arrays(result)
