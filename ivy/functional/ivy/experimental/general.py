# global
import functools
from typing import Callable, Union, Sequence

# local
import ivy
from ivy import to_ivy_arrays_and_back
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@inputs_to_ivy_arrays
def reduce(
    operand: Union[ivy.Array, ivy.NativeArray],
    init_value: Union[int, float],
    func: Callable,
    axes: Union[int, Sequence[int]] = 0,
    keepdims: bool = False,
) -> ivy.Array:
    """
    Reduces the input array's dimensions by applying a function along one or more axes.

    Parameters
    ----------
    operand
        The array to act on.
    init_value
        The value with which to start the reduction.
    func
        The reduction function.
    axes
        The dimensions along which the reduction is performed.
    keepdims
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.

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
    axes = (axes,) if isinstance(axes, int) else axes
    axes = sorted(axes, reverse=True)
    axes = [a + operand.ndim if a < 0 else a for a in axes]
    init_value = ivy.array(init_value)
    op_dtype = operand.dtype
    for axis in axes:
        temp = ivy.moveaxis(operand, axis, 0).reshape((operand.shape[axis], -1))
        temp = functools.reduce(func, temp, init_value)
        operand = ivy.reshape(temp, operand.shape[:axis] + operand.shape[axis+1:])
    if keepdims:
        operand = ivy.expand_dims(operand, axis=axes)
    return operand.astype(op_dtype)
