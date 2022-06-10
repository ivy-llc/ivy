"""Collection of Ivy activation functions."""

from typing import Union, Optional

# local
import ivy
from ivy.array.conversions import to_native
from ivy.backend_handler import current_backend as _cur_backend
from ivy.func_wrapper import handle_out_argument, to_native_arrays_and_back


# Extra #
# ------#


@to_native_arrays_and_back
@handle_out_argument
def relu(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
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
    return _cur_backend(x).relu(x, out)


@to_native_arrays_and_back
@handle_out_argument
def leaky_relu(
    x: Union[ivy.Array, ivy.NativeArray], alpha: Optional[float] = 0.2
) -> ivy.Array:
    """Applies the leaky rectified linear unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for ReLU.

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
    return _cur_backend(x).leaky_relu(x, alpha)

<<<<<<< HEAD
=======

@to_native_arrays_and_back
@handle_out_argument
>>>>>>> cde340fc8199e9da5d371eef8cdafcb1f61b866d
def gelu(x: Union[ivy.Array, ivy.NativeArray], approximate: bool = True)\
    -> ivy.Array:
    """
    Applies the Gaussian error linear unit (GELU) activation function.

    Parameters
    -----------
    x: 
        Input array.
    approximate: 
        Whether to approximate. Default: True.

    Returns
    -------

    out:  
        The input array with gelu applied element-wise on ``x``.

    Examples:
    >>> x = ivy.array([-1. , 0. , 1. ])
    >>> y = ivy.gelu(x, True)
    >>> print(y)
    [-0.158808, 0. , 0.841192]

    >>> x = ivy.array([-1. , 0. , 1. ])
    >>> y = ivy.gelu(x, False)
    >>> print(y)
    [-0.15865526,  0. , 0.8413447]

    """
    return _cur_backend(x).gelu(x, approximate)


@to_native_arrays_and_back
@handle_out_argument
def tanh(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """Applies the Hyperbolic tangent activation function element-wise.

    Parameters
    ----------
    x
        input array

    Returns
    -------
    ret
        The input array with Hyperbolic tangent activation applied element-wise.


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

    Instance Method Example
    -----------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([0.55 , -0.55])
    >>> y = x.tanh()
    >>> print(y)
    ivy.array([0.501, -0.501])

    """
    return _cur_backend(x).tanh(x)


@to_native_arrays_and_back
@handle_out_argument
def sigmoid(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """Applies the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.

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
    return _cur_backend(x).sigmoid(x)


@to_native_arrays_and_back
@handle_out_argument
def softmax(
    x: Union[ivy.Array, ivy.NativeArray], axis: Optional[int] = -1
) -> ivy.Array:
    """Applies the softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension softmax would be performed on. The default is -1 which indicates
        the last dimension.

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
    return _cur_backend(x).softmax(x, axis)


@to_native_arrays_and_back
@handle_out_argument
def softplus(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """Applies the softplus function element-wise.

    Parameters
    ----------
    x
        input array.

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
    ivy.array([0.5349962, 0.4203641])

    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x)
    >>> print(y)
    ivy.array([0.5349962, 0.4203641])

    Instance Method Example
    ------------------------

    Using :code: `ivy.Array` instance method:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = x.softplus()
    >>> print(y)
    ivy.array([0.5349962, 0.4203641])

    """
    return _cur_backend(x).softplus(x)
