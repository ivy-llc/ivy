"""Collection of Ivy activation functions."""

from typing import Union, Optional

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
)


# Extra #
# ------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def relu(
    x: Union[ivy.Array, ivy.NativeArray ivy.Container], *, out: Optional[ivy.Array, ivy.NativeArray ivy.Container] = None
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
        The ReLU : element-wise max(x, 0).

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

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([-4.5, -3, 1, 1])
    >>> y = ivy.native_array([0.1, 0.2, 0.8, -0.6])
    >>> z = ivy.relu(x, out=y)
    >>> print(z)
    ivy.array([0.1, 0.2, 0.8, 0.])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -6.0, 0.9]),b=ivy.array([-10.0, 3.0, 1.0]))
    >>> y = ivy.Container(a=ivy.array([0.6, 0.2, 0.3]),b=ivy.array([0.8, 0.2, -0.2]))
    >>> z = ivy.relu(x, y)
    >>> print(z)
    {a:ivy.array([0.6, 0.2, 0.3]), b:ivy.array([0.8, 0.2, 0.])}

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([1.0 , -3.2, -1.0])
    >>> y = ivy.Container(a=ivy.array([0.7, -0.8, 0.2]))
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    {
       a: ivy.array([0.7, 0., 0.2])
    }

    Instance Method Examples
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-0.5, 1., -2.5])
    >>> y = x.relu()
    >>> print(y)
    ivy.array([0., 1., 0.])

    """
    return ivy.current_backend(x).relu(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def leaky_relu(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    alpha: Optional[float] = 0.2,
    *,
    out: Optional[ivy.Array, ivy.NativeArray ivy.Container] = None,
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
        The LeakyReLU : ((x > 0) * x) + ((x <= 0) * x * 0.01)

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

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.native_array([0.1, -0.2, 0.8])
    >>> z = ivy.leaky_relu(x, y)
    >>> print(z)
    ivy.array([0.1, -0.16, 0.8])

     With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -9.5, 0]),b=ivy.array([-1.1, 0, 1]))
    >>> y = ivy.Container(a=ivy.array([0.6, 0.2, -0.3]),b=ivy.array([0.8, -0.2, 0.2]))
    >>> z = ivy.leaky_relu(x, y)
    >>> print(z)
    {a:ivy.array([0.6, 0.2, -0.24]),b:ivy.array([0.8, -0.16, 0.2])}

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([1 , -1, 0])
    >>> y = ivy.Container(a=ivy.array([-0.7, 0.8, 0.2]))
    >>> z = ivy.leaky_relu(x, y)
    >>> print(z)
    {
       a: ivy.array([-0.56, 0.8, 0.2])
    }

    Instance Method Examples
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-0.5, 1., -2.5])
    >>> y = x.leaky_relu()
    >>> print(y)
    ivy.array([-0.1,  1. , -0.5])

    """
    return ivy.current_backend(x).leaky_relu(x, alpha, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def gelu(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    approximate: True,
    *,
    out: Optional[ivy.Array, ivy.NativeArray ivy.Container] = None,
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
        The gaussian error linear activation: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))) if approximate is True or
        x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2))), where P(X) ~ N(0, 1), if approximate is False.

    Functional Examples
    -------------------

    With :code: `ivy.Array` input:
    For approximate: False
    >>> x = ivy.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    >>> y = ivy.gelu(x)
    >>> print(y)
    ivy.array([-0.00404951, -0.15865529, 0. , 0.8413447, 2.9959507])

    For approximate: True
    >>> x = ivy.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    >>> y = ivy.gelu(x)
    >>> print(y)
    ivy.array([-0.00363752, -0.15880796, 0., 0.841192, 2.9963627])

    For approximate: True
    >>> x = ivy.array([[-3.0, -1.0, 0.0, 1.0, 3.0],[-2.0, -1.0, 0.0, 1.0, 2.0]])
    >>> ivy.gelu(x, out = x)
    >>> print(x)
    ivy.array([[-3., -1., 0., 1., 3.],
               [-2., -1., 0., 1., 2.]])


    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([0., -1., 2.])
    >>> y = ivy.gelu(x)
    >>> print(y)
    ivy.array([ 0. , -0.2,  2. ])

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1.0, 0.7, -2.4])
    >>> y = ivy.native_array([0.1, 0.2, -0.8])
    >>> z = ivy.gelu(x, y)
    >>> print(z)
    ivy.array([0.05, 0.11, -0.16])

     With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -9.5, 0]),b=ivy.array([-1.1, 0, 1]))
    >>> y = ivy.Container(a=ivy.array([0.6, 0.2, -0.3]),b=ivy.array([1.0, -9.5, 0]))
    >>> z = ivy.gelu(x, y)
    >>> print(z)
    {a:ivy.array([0.435, 0.115, -0.114]),b:ivy.array([0.841, 0., 0.])}

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([1 , -1, 0])
    >>> y = ivy.Container(a=ivy.array([0.1, 0.2, -0.8]))
    >>> z = ivy.gelu(x, y)
    >>> print(z)
    {
       a: ivy.array([0.05, 0.11, -0.16])
    }

    Instance Method Examples
    ------------------------

    Using :code: `ivy.Array` instance method:
    For approximate: True
    >>> x = ivy.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    >>> y = ivy.gelu()
    >>> print(y)
    ivy.array([-0.00363752, -0.15880796, 0., 0.841192, 2.9963627])
    """
    return ivy.current_backend(x).gelu(x, approximate, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def tanh(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container], *, out: Optional[ivy.Array, ivy.NativeArray ivy.Container] = None
) -> ivy.Array:
    """Applies the Hyperbolic tangent activation function element-wise.

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
        The input array with Hyperbolic tangent activation applied element-wise.
        The Tanh: Tensor of same shape and dtype of input x, with tanh activation: tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x))).

    Functional Examples
    -------------------

    With :code: `ivy.Array` input:

    >>> x = ivy.array([0.55 , -0.55])
    >>> y = ivy.tanh(x)
    >>> print(y)
    ivy.array([0.501, -0.501])

    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([0., -1., 2.])
    >>> y = ivy.tanh(x)
    >>> print(y)
    ivy.array([0., -0.762, 0.964])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -9.5, 0]),b=ivy.array([-1.1, 0, 1.0]))
    >>> z = ivy.tanh(x)
    >>> print(z)
    {a:ivy.array([0.76, -0.99, 0.]),b:ivy.array([-0.80, 0., 0.76])

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1.0, 0.7, -2.4])
    >>> y = ivy.native_array([0.1, 0.2, -0.8])
    >>> z = ivy.tanh(x, y)
    >>> print(z)
    ivy.array([0.099, 0.19, -0.66])

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([0.1, 0.2, -0.8])
    >>> y = ivy.Container(a=ivy.array([1 , -1, 0]))
    >>> z = ivy.tanh(x, y)
    >>> print(z)
    {
       a: ivy.array([0.76, -0.76, 0.])
    }


    Instance Method Example
    -----------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([0.55 , -0.55])
    >>> y = x.tanh()
    >>> print(y)
    ivy.array([0.501, -0.501])

    """
    return ivy.current_backend(x).tanh(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def sigmoid(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container], *, out: Optional[ivy.Array, ivy.NativeArray ivy.Container] = None
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
        The Sigmoid: Tensor with the sigmoid activation: 1 / (1 + exp(-x)).

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

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -9.5, 0]),b=ivy.array([-1.1, 0, 1.0]))
    >>> z = ivy.sigmoid(x)
    >>> print(z)
    {a:ivy.array([7.310586e-01, 7.485183e-05, 5.000000e-01]),b:ivy.array([0.24973989, 0.5, 0.7310586 ])

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1.0, 0.7, -2.4])
    >>> y = ivy.native_array([0.1, 0.2, -0.8])
    >>> z = ivy.sigmoid(x, y)
    >>> print(z)
    ivy.array([0.52, 0.54, 0.31])

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([0.1, 0.2, -0.8])
    >>> y = ivy.Container(a=ivy.array([1 , -1, 0]))
    >>> z = ivy.tanh(x, y)
    >>> print(z)
    {
       a: ivy.array([0.73, 0.26, 0.5])
    }

    Instance Method Example
    -----------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-1., 1., 2.])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    """
    return ivy.current_backend(x).sigmoid(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def softmax(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    axis: Optional[int] = -1,
    *,
    out: Optional[ivy.Array, ivy.NativeArray ivy.Container] = None,
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
        The Softmax: exp(x) / tf.reduce_sum(exp(x)).

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

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -9.5, 0]),b=ivy.array([-1.1, 0, 1.0]))
    >>> z = ivy.softmax(x,axis=1)
    >>> print(z)
    {a:ivy.array([7.31043862e-01, 2.01303523e-05, 2.68936007e-01]),b:ivy.array([8.21670006e-02, 2.46843311e-01, 6.70989688e-01])

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1.0, 0.7, -2.4])
    >>> y = ivy.native_array([-0.8, 0, 1.0])
    >>> z = ivy.softmax(x, y, axis=0)
    >>> print(z)
    ivy.array([0.10, 0.23, 0.65])

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([0.1, 0.2, -0.8])
    >>> y = ivy.Container(a=ivy.array([1 , -1, 0]))
    >>> z = ivy.softmax(x, y, axis=0)
    >>> print(z)
    {
       a: ivy.array([0.66, 0.09, 0.24])
    }

    Instance Method Example
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([1.0, 0, 1.0])
    >>> y = x.softmax()
    >>> print(y)
    ivy.array([0.422, 0.155, 0.422])

    """
    return ivy.current_backend(x).softmax(x, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def softplus(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    *,
    out: Optional[ivy.Array, ivy.NativeArray ivy.Container] = None
) -> ivy.Array:
    """Applies the softplus function element-wise.

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
        an array containing the softplus activation of each element in ``x``.

    Functional Examples
    -------------------

    With :code: `ivy.Array` input:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x)
    >>> print(y)
    ivy.array([0.535,0.42])


    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x)
    >>> print(y)
    ivy.array([0.535,0.42])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -9.5, 0]),b=ivy.array([-1.1, 0, 1.0]))
    >>> z = ivy.softplus(x,axis=1)
    >>> print(z)
    {a:ivy.array([1.3132616e+00, 7.4849027e-05, 6.9314718e-01]),b:ivy.array([2.8733534e-01, 6.9314718e-01, 1.3132616e+00])

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1.0, 0.7, -2.4])
    >>> y = ivy.native_array([-0.8, 0, 1.0])
    >>> z = ivy.softplus(x, y)
    >>> print(z)
    ivy.array([0.37110066, 0.6931472 , 1.3132616])

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([0.1, 0.2, -0.8])
    >>> y = ivy.Container(a=ivy.array([1 , -1, 0]))
    >>> z = ivy.softplus(x, y)
    >>> print(z)
    {
       a: ivy.array([1.3132616, 0.3132617, 0.6931472])
    }

    Instance Method Example
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = x.softplus()
    >>> print(y)
    ivy.array([0.535,0.42])

    """
    return ivy.current_backend(x).softplus(x, out=out)
