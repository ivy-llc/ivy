# global
import functools
from typing import Callable, Union, Sequence

# local
import ivy
from ivy import (
    inputs_to_ivy_arrays,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_array_function,
)
from ivy.utils.exceptions import handle_exceptions


def _correct_ivy_callable(func):
    # get the current backend of the given ivy callable
    if ivy.nested_any(
        func,
        lambda x: hasattr(x, "__module__")
        and x.__module__.startswith("ivy")
        and not x.__module__.startswith("ivy.functional.frontends"),
    ):
        return ivy.__dict__[func.__name__]
    return func


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def reduce(
    operand: Union[ivy.Array, ivy.NativeArray],
    init_value: Union[int, float],
    computation: Callable,
    /,
    *,
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
    computation
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
    axes = (axes,) if isinstance(axes, int) else axes
    axes = [a + operand.ndim if a < 0 else a for a in axes]
    axes = sorted(axes, reverse=True)
    init_value = ivy.array(init_value)
    op_dtype = operand.dtype
    computation = _correct_ivy_callable(computation)
    for axis in axes:
        temp = ivy.moveaxis(operand, axis, 0).reshape((operand.shape[axis], -1))
        temp = functools.reduce(computation, temp, init_value)
        operand = ivy.reshape(temp, operand.shape[:axis] + operand.shape[axis + 1 :])
    if keepdims:
        operand = ivy.expand_dims(operand, axis=axes)
    return operand.astype(op_dtype)


reduce.mixed_backend_wrappers = {
    "to_add": (
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}
