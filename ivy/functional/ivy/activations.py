"""Collection of Ivy activation functions."""

from typing import Union, Optional

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    integer_arrays_to_float,
)
from ivy.exceptions import handle_exceptions


# Extra #
# ------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def relu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Applies the rectified linear unit function element-wise.

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

    Functional Examples
    -------------------

    With :code: `ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.relu(x)
    >>> print(y)
    ivy.array([0., 0., 1.])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.relu(x, out = y)
    >>> print(y)
    ivy.array([1.5, 0.7, 0.])

    >>> x = ivy.array([[1.1, 2.2, 3.3], \
                       [-4.4, -5.5, -6.6]])
    >>> ivy.relu(x, out = x)
    >>> print(x)
    ivy.array([[1.1, 2.2, 3.3],
               [0., 0., 0.]])

    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([0., -1., 2.])
    >>> y = ivy.relu(x)
    >>> print(y)
    ivy.array([0., 0., 2.])

    Instance Method Examples
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-0.5, 1., -2.5])
    >>> y = x.relu()
    >>> print(y)
    ivy.array([0., 1., 0.])

    """
    return current_backend(x).relu(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def leaky_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 0.2,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the leaky rectified linear unit function element-wise.

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

    Functional Examples
    -------------------

    With :code: `ivy.Array` input:

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.leaky_relu(x)
    >>> print(y)
    ivy.array([ 0.39, -0.17])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.leaky_relu(x, out = y)
    >>> print(y)
    ivy.array([ 1.5 ,  0.7 , -0.48])

    >>> x = ivy.array([[1.1, 2.2, 3.3], \
                       [-4.4, -5.5, -6.6]])
    >>> ivy.leaky_relu(x, out = x)
    >>> print(x)
    ivy.array([[ 1.1 ,  2.2 ,  3.3 ],
       [-0.88, -1.1 , -1.32]])


    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([0., -1., 2.])
    >>> y = ivy.leaky_relu(x)
    >>> print(y)
    ivy.array([ 0. , -0.2,  2. ])


    Instance Method Examples
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-0.5, 1., -2.5])
    >>> y = x.leaky_relu()
    >>> print(y)
    ivy.array([-0.1,  1. , -0.5])

    """
    return current_backend(x).leaky_relu(x, alpha=alpha, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def gelu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    approximate=True,
    out: Optional[ivy.Array] = None,
):
    """Applies the Gaussian error linear unit (GELU) activation function.

    Parameters
    ----------
    x
        Input array.
    approximate
        Whether to approximate, default is True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with gelu applied element-wise.

    """
    return current_backend(x).gelu(x, approximate=approximate, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Applies the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.

    Functional Examples
    -------------------

    With :code: `ivy.Array` input:

    >>> x = ivy.array([-1., 1., 2.])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([-1.3, 3.8, 2.1])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.214, 0.978, 0.891])

    Instance Method Example
    -----------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-1., 1., 2.])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    """
    return current_backend(x).sigmoid(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def softmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension softmax would be performed on. The default is -1 which indicates
        the last dimension.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with softmax applied element-wise.

    Functional Examples
    -------------------

    With :code: `ivy.Array` input:

    >>> x = ivy.array([1.0, 0, 1.0])
    >>> y = ivy.softmax(x)
    >>> print(y)
    ivy.array([0.422, 0.155, 0.422])

    >>> x = ivy.array([[1.1, 2.2, 3.3], \
                       [4.4, 5.5, 6.6]])
    >>> y = ivy.softmax(x, axis = 1)
    >>> print(y)
    ivy.array([[0.0768, 0.231 , 0.693 ],
               [0.0768, 0.231 , 0.693 ]])


    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([1.5, 0.3, 1.2])
    >>> y = ivy.softmax(x)
    >>> print(y)
    ivy.array([0.49 , 0.147, 0.363])

    Instance Method Example
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([1.0, 0, 1.0])
    >>> y = x.softmax()
    >>> print(y)
    ivy.array([0.422, 0.155, 0.422])

    """
    return current_backend(x).softmax(x, axis=axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def softplus(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the softplus function element-wise.

    Parameters
    ----------
    x
        input array.
    beta
        The beta value for the softplus formation. Default: None.
    threshold
        values above this revert to a linear function. Default: None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the softplus activation of each element in ``x``.

    Functional Examples
    -------------------

    With :code: `ivy.Array` input:

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

    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x)
    >>> print(y)
    ivy.array([0.535,0.42])


    Instance Method Example
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = x.softplus()
    >>> print(y)
    ivy.array([0.535,0.42])

    """
    return current_backend(x).softplus(x, beta=beta, threshold=threshold, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def log_softmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the log_softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension softmax would be performed on. The default is -1 which indicates
        the last dimension.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The output array with log_softmax applied element-wise to input.
    """
    return current_backend(x).log_softmax(x, axis=axis, out=out)
