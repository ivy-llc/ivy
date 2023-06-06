"""Collection of Ivy activation functions."""

from typing import Union, Optional, Callable
import sys

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_function,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    integer_arrays_to_float,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions


# Extra #
# ------#


@handle_exceptions
def deserialize(
    name: Union[str, None], /, *, custom_objects: Optional[ivy.Dict] = None
) -> Union[Callable, None]:
    """
    Return activation function given a string identifier.

    Parameters
    ----------
    name
        The name of the activation function.
    custom_objects
        Optional dictionary listing user-provided activation functions.

    Returns
    -------
    ret
        Corresponding activation function.

    Examples
    --------
    With :code:`str` input:

    >>> name = "sigmoid"
    >>> sigmoid = ivy.deserialize(name)
    >>> print(sigmoid)
    <function sigmoid at XXXXXXXXXXXXXX>

    With :code:`str` and :code:`dict` input:

    >>> name = "custom_fn"
    >>> objects = {"custom_fn": lambda x: x}
    >>> custom_fn = ivy.deserialize(name, custom_objects=objects)
    >>> print(custom_fn)
    <function custom_fn at XXXXXXXXXXXXXX>
    """
    if current_backend().__name__.split(".")[-1] == "tensorflow":
        return current_backend().deserialize(name, custom_objects=custom_objects)

    if name is None:
        return None

    module_name = "ivy.functional.ivy.activations"
    activation_functions = {}
    module = sys.modules[module_name]

    for fn_name in dir(module):
        obj = getattr(module, fn_name)
        if callable(obj) and fn_name in ACTIVATION_FUNCTIONS:
            activation_functions[fn_name] = obj

    if isinstance(name, str):
        if custom_objects and name in custom_objects:
            fn_obj = custom_objects.get(name)
        else:
            fn_obj = activation_functions.get(name)
            if fn_obj is None:
                raise ValueError(f"Unknown activation function: {name}.")
        return fn_obj

    else:
        raise ValueError(f"Could not interpret serialized activation function: {name}")


ACTIVATION_FUNCTIONS = [
    "gelu",
    "leaky_relu",
    "log_softmax",
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
]


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@integer_arrays_to_float
@handle_array_function
def gelu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    approximate: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the Gaussian error linear unit (GELU) activation function.

    Parameters
    ----------
    x
        Input array.
    approximate
        Whether to approximate, default is ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with gelu applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.2, -0.6, 1.5])
    >>> y = ivy.gelu(x)
    >>> y
    ivy.array([-0.138, -0.165, 1.4])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([-1.3, 3.8, 2.1])
    >>> y = ivy.gelu(x)
    >>> y
    ivy.array([-0.126, 3.8, 2.06])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1., 2.]), b=ivy.array([-0.9, -1.]))
    >>> y = ivy.gelu(x)
    >>> y
    {
        a: ivy.array([0.841, 1.95]),
        b: ivy.array([-0.166, -0.159])
    }
    """
    return current_backend(x).gelu(x, approximate=approximate, out=out)


@handle_exceptions
def get(
    name: Union[str, None], /, *, custom_objects: Optional[ivy.Dict] = None
) -> Union[Callable, None]:
    """
    Return activation function given a string identifier.

    Parameters
    ----------
    name
        The name of the activation function.
    custom_objects
        Optional dictionary listing user-provided activation functions.

    Returns
    -------
    ret
        Corresponding activation function.

    Examples
    --------
    With :code:`str` input:

    >>> name = "sigmoid"
    >>> sigmoid = ivy.get(name)
    >>> print(sigmoid)
    <function sigmoid at XXXXXXXXXXXXXX>

    >>> name = None
    >>> linear = ivy.get(name)
    >>> print(linear)
    <function linear at XXXXXXXXXXXXXX>

    With :code:`str` and :code:`dict` input:

    >>> name = "custom_fn"
    >>> objects = {"custom_fn": lambda x: x}
    >>> custom_fn = ivy.get(name, custom_objects=objects)
    >>> print(custom_fn)
    <function custom_fn at XXXXXXXXXXXXXX>
    """
    if current_backend().__name__.split(".")[-1] == "tensorflow":
        return current_backend().get(name, custom_objects=custom_objects)

    if name is None:
        return ivy.linear

    return ivy.deserialize(name, custom_objects=custom_objects)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def leaky_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 0.2,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the leaky rectified linear unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for ReLU.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with leaky relu applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.leaky_relu(x)
    >>> print(y)
    ivy.array([ 0.39, -0.17])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.leaky_relu(x, out=y)
    >>> print(y)
    ivy.array([ 1.5 ,  0.7 , -0.48])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.leaky_relu(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1 ,  2.2 ,  3.3 ],
       [-0.88, -1.1 , -1.32]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.leaky_relu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([0., -0.24000001]),
        b: ivy.array([0.40000001, -0.04])
    }
    """
    return current_backend(x).leaky_relu(x, alpha=alpha, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def log_softmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the log_softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension log_softmax would be performed on. The default is ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The output array with log_softmax applied element-wise to input.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, -0.98])
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    ivy.array([-0.703, -0.683])

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    ivy.array([-2.41, -1.41, -0.408])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1.5, 0.5, 1.0])
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    ivy.array([-0.68, -1.68, -1.18])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.5, 0.5, 1.0]))
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    {
        a: ivy.array([-0.68, -1.68, -1.18])
    }

    >>> x = ivy.Container(a=ivy.array([1.0, 2.0]), b=ivy.array([0.4, -0.2]))
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    {
        a: ivy.array([-1.31, -0.313]),
        b: ivy.array([-0.437, -1.04])
    }
    """
    return current_backend(x).log_softmax(x, axis=axis, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def relu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the rectified linear unit function element-wise.

    Parameters
    ----------
    x
        input array
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rectified linear unit activation of each element in
        ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.relu(x)
    >>> print(y)
    ivy.array([0., 0., 1.])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.relu(x, out = y)
    >>> print(y)
    ivy.array([1.5, 0.7, 0.])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.relu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([1., 0.]),
        b: ivy.array([0.40000001, 0.])
    }
    """
    return current_backend(x).relu(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@integer_arrays_to_float
@handle_array_function
def sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([[0.214, 0.978, 0.891], [0.846,0.985,0.001]] )
    """
    return current_backend(x).sigmoid(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def softmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension softmax would be performed on. The default is ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with softmax applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1.0, 0, 1.0])
    >>> y = ivy.softmax(x)
    >>> print(y)
    ivy.array([0.422, 0.155, 0.422])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [4.4, 5.5, 6.6]])
    >>> y = ivy.softmax(x, axis = 1)
    >>> print(y)
    ivy.array([[0.0768, 0.231 , 0.693 ],
               [0.0768, 0.231 , 0.693 ]])
    """
    return current_backend(x).softmax(x, axis=axis, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def softplus(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the softplus function element-wise.

    Parameters
    ----------
    x
        input array.
    beta
        The beta value for the softplus formation. Default: ``None``.
    threshold
        values above this revert to a linear function. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the softplus activation of each element in ``x``.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x)
    >>> print(y)
    ivy.array([0.535,0.42])

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x, beta=0.5)
    >>> print(y)
    ivy.array([1.22, 1.09])

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.softplus(x, threshold=2)
    >>> print(y)
    ivy.array([1.31, 2.13, 3.  ])
    """
    return current_backend(x).softplus(x, beta=beta, threshold=threshold, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def mish(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the mish activation function element-wise.

    Parameters
    ----------
    x
        input array
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the mish activation of each element in
        ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.mish(x)
    >>> print(y)
    ivy.array([-0.30340147,  0.        ,  0.86509842])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.mish(x, out = y)
    >>> print(y)
    ivy.array([ 1.40337825,  0.56114835, -0.20788449])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.mish(x)
    >>> print(x)
    {
        a: ivy.array([0.86509842, -0.30883577]),
        b: ivy.array([0.28903052, -0.10714479])
    }
    """
    return current_backend(x).mish(x, out=out)
