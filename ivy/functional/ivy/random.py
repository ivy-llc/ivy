"""Collection of random Ivy functions."""

# global
from typing import Optional, Union

# local
import ivy
from ivy.func_wrapper import (
    infer_dtype,
    infer_device,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
)
from ivy.backend_handler import backend_stack


# Helpers #
# ------- #


def _check_bounds_and_get_shape(low, high, shape):
    if shape is not None:
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise Exception(
                "`shape` argument can only be specified when `low` \
                              and `high` arguments are numerics (not arrays)"
            )
        return shape
    valid_types = (ivy.Array, ivy.NativeArray)
    if len(backend_stack) == 0:
        valid_types += (ivy.current_backend().NativeArray,)
    if isinstance(low, valid_types):
        if isinstance(high, valid_types):
            if ivy.shape(low) != ivy.shape(high):
                raise Exception("shape of bounds have to be the same")
        return ivy.shape(low)
    if isinstance(high, valid_types):
        return ivy.shape(high)
    return ()


def _check_valid_scale(std):
    if (isinstance(std, (int, float)) and std < 0) or ivy.any(ivy.less(std, 0)):
        raise Exception("`std` must be non-negative")


# Extra #
# ------#


@to_native_arrays_and_back
@handle_out_argument
@infer_device
@infer_dtype
@handle_nestable
def random_uniform(
    low: Union[float, ivy.NativeArray, ivy.Array] = 0.0,
    high: Union[float, ivy.NativeArray, ivy.Array] = 1.0,
    shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Draws samples from a uniform distribution. Samples are uniformly distributed over
    the half-open interval ``[low, high)`` (includes ``low``, but excludes ``high``). In
    other words, any value within the given interval is equally likely to be drawn by
    uniform.

    Parameters
    ----------
    low
        Lower boundary of the output interval. All values generated will be greater than
        or equal to ``low``. If array, must have same shape as ``high``.
    high
        Upper boundary of the output interval. All the values generated will be less
        than ``high``. If array, must have same shape as ``low``.
    shape
        If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples are drawn.
        Can only be specified when ``low`` and ``high`` are numeric values, else
        exception will be raised.
        Default is ``None``, where a single value is returned.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None).
    dtype
         output array data type. If ``dtype`` is ``None``, the output array data
         type will be the default floating-point data type. Default ``None``
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Drawn samples from the parameterized uniform distribution.

    Functional Examples
    -------------------

    >>> ivy.random_uniform()
    ivy.array(0.26431865)

    >>> ivy.random_uniform(shape=3)
    ivy.array([0.475, 0.878, 0.861])

    >>> ivy.random_uniform(shape=(2,3))
    ivy.array([[0.929 , 0.545 , 0.789 ],
               [0.519 , 0.0435, 0.381 ]])

    >>> ivy.random_uniform(3.0, 6.0)
    ivy.array(3.4608004)

    >>> ivy.random_uniform(1.0, 2.0, (2,1))
    ivy.array([[1.85],
               [1.81]])

    >>> z = ivy.zeros(())
    >>> ivy.random_uniform(1.0, 2.0, out=z)
    ivy.array(1.8458502)

    >>> ivy.random_uniform(1.0, 2.0, (2,2), device='cpu')
    ivy.array([[1.81, 1.8 ],
               [1.32, 1.43]])

    >>> ivy.random_uniform(1.0, 2.0, (2,2), device='cpu', dtype='int32')
    ivy.array([[1, 1],
               [1, 1]])

    >>> z = ivy.zeros((1,2))
    >>> ivy.random_uniform(1.0, 2.0, (1,2), device='cpu', dtype='float64', out=z)
    ivy.array([[1.34, 1.02]])

    >>> x = ivy.array([4.8, 5.6])
    >>> y = ivy.array([9.8, 7.4])
    >>> ivy.random_uniform(x, y)
    ivy.array([0.475, 0.878])

    >>> z = ivy.zeros((2,))
    >>> ivy.random_uniform(x, y, out=z)
    ivy.array([9.41, 7.17])

    >>> ivy.random_uniform(x, y, device='cpu')
    ivy.array([6.88, 6.75])

    >>> ivy.random_uniform(x, y, device='cpu', dtype='float64')
    ivy.array([8.62, 6.47])

    >>> z = ivy.zeros((2,))
    >>> ivy.random_uniform(x, y, device='cpu', dtype='float64', out=z)
    ivy.array([5. , 7.3])
    """
    return ivy.current_backend().random_uniform(
        low, high, shape, device=device, dtype=dtype, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@infer_device
@infer_dtype
@handle_nestable
def random_normal(
    mean: Union[float, ivy.NativeArray, ivy.Array] = 0.0,
    std: Union[float, ivy.NativeArray, ivy.Array] = 1.0,
    shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.array:
    """
    Draws samples from a normal distribution.

    Parameters
    ----------
    mean
        The mean of the normal distribution to sample from. Default is ``0.0``.
    std
        The standard deviation of the normal distribution to sample from.
        Must be non-negative. Default is ``1.0``.
    shape
        If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples are drawn.
        Can only be specified when ``mean`` and ``std`` are numeric values, else
        exception will be raised.
        Default is ``None``, where a single value is returned.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating-point data type. Default ``None``
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None).
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
     ret
        Drawn samples from the parameterized normal distribution.

    Functional Examples
    -------------------

    >>> ivy.random_normal()
    ivy.array(-0.22346112)

    >>> ivy.random_normal(shape=3)
    ivy.array([-0.73  ,  0.0922, -0.515 ])

    >>> ivy.random_normal(shape=(2,3))
    ivy.array([[-0.361 ,  0.596 , -0.247 ],
               [-1.39  ,  0.0426, -0.627 ]])

    >>> ivy.random_normal(3.0, 6.0)
    ivy.array(4.9213753)

    >>> ivy.random_normal(1.0, 2.0, (2,1))
    ivy.array([[2.19],
               [2.78]])

    >>> z = ivy.zeros(())
    >>> ivy.random_normal(1.0, 2.0, out=z)
    ivy.array(0.12818667)

    >>> ivy.random_normal(1.0, 2.0, (2,2), device='cpu')
    ivy.array([[ 2.91 ,  1.3  ],
               [ 3.37 , -0.799]])

    >>> ivy.random_normal(1.0, 2.0, (2,2), device='cpu', dtype='int32')
    ivy.array([[ 0, -1],
               [ 0,  3]])

    >>> z = ivy.zeros((1,2))
    >>> ivy.random_normal(1.0, 2.0, (1,2), device='cpu', dtype='float64', out=z)
    ivy.array([[-2.01, -1.95]])

    >>> x = ivy.array([4.8, 5.6])
    >>> y = ivy.array([9.8, 7.4])
    >>> ivy.random_normal(x, y)
    ivy.array([ 4.43 , -0.469])

    >>> z = ivy.zeros((2,))
    >>> ivy.random_normal(x, y, out=z)
    ivy.array([0.287, 8.55 ])

    >>> ivy.random_normal(x, y, device='cpu')
    ivy.array([18.9, 15.2])

    >>> ivy.random_normal(x, y, device='cpu', dtype='float64')
    ivy.array([-4.1   , -0.0366])

    >>> z = ivy.zeros((2,))
    >>> ivy.random_normal(x, y, device='cpu', dtype='float64', out=z)
    ivy.array([12.4, 11. ])

    Instance Method Examples
    ------------------------
    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([7.5,6.7,0.9]), b=ivy.array([8.7,9.8,4.5]))
    >>> x.random_normal(17.4)
    {
        a: ivy.array([14.4, 10.5, 10.5]),
        b: ivy.array([8.93, 12.1, 17.3])
    }

    >>> x.random_normal(10.2, device='cpu')
    {
        a: ivy.array([9.32, 7.4, 4.62]),
        b: ivy.array([8.73, 10., 9.97])
    }

    >>> x.random_normal(14.2, dtype='float16')
    {
        a: ivy.array([9.26, 8.79, 3.34]),
        b: ivy.array([11.3, 11.9, 12.2])
    }

    >>> x.random_normal(10.8, device='cpu', dtype='float64')
    {
        a: ivy.array([8.91, 6.99, 7.75]),
        b: ivy.array([9.35, 10.5, 8.5])
    }

    >>> z = ivy.Container(a=ivy.zeros((3,)), b=ivy.ones((3,)))
    >>> x.random_normal(11.2, device='cpu', dtype='float64', out=z)
    {
        a: ivy.array([10.8, 7.6, 2.48]),
        b: ivy.array([10.1, 10.1, 6.97])
    }

    >>> y = ivy.Container(a=10.4, b=17.4)
    >>> x.random_normal(y)
    {
        a: ivy.array([10., 8.38, 4.64]),
        b: ivy.array([15.4, 10.3, 12.7])
    }

    >>> x.random_normal(y, device='cpu')
    {
        a: ivy.array([8.47, 8.23, 8.69]),
        b: ivy.array([10.7, 16.2, 16.1])
    }

    >>> x.random_normal(y, dtype='float16')
    {
        a: ivy.array([7.55, 10.3, 1.57]),
        b: ivy.array([10.8, 14., 14.1])
    }

    >>> x.random_normal(y, device='cpu', dtype='float64')
    {
        a: ivy.array([8.97, 6.91, 6.17]),
        b: ivy.array([13.5, 14., 10.9])
    }

    >>> z = ivy.Container(a=ivy.zeros((3,)), b=ivy.ones((3,)))
    >>> x.random_normal(y, device='cpu', dtype='float64', out=z)
    {
        a: ivy.array([7.7, 9.37, 9.08]),
        b: ivy.array([17.1, 14.5, 16.7])
    }

    >>> x = ivy.Container(a=ivy.array([[9.8,7.6],[6.5,2.3]]), \
                          b=ivy.array([[0.9,2.4],[7.6,5.4]]))
    >>> y = ivy.Container(a=ivy.array([[10.9,32.4],[18.7,19.6]]), \
                          b=ivy.array([[4.3,5.6],[23.4,54.3]]))
    >>> x.random_normal(y)
    {
        a: ivy.array([[10.5, 11.7],
                      [8.59, 5.6]]),
        b: ivy.array([[1.4, 5.17],
                      [9.1, 19.4]])
    }

    >>> x.random_normal(y, device='cpu')
    {
        a: ivy.array([[10., 29.4],
                      [12.5, 11.4]]),
        b: ivy.array([[2.22, 4.84],
                      [9.52, 31.8]])
    }

    >>> x.random_normal(y, dtype='float16')
    {
        a: ivy.array([[10.5, 31.1],
                      [8.62, 10.7]]),
        b: ivy.array([[4.14, 3.48],
                      [16.8, 45.9]])
    }

    >>> x.random_normal(y, device='cpu', dtype='float64')
    {
        a: ivy.array([[10.4, 15.9],
                      [13.9, 12.5]]),
        b: ivy.array([[3.61, 3.96],
                      [20.3, 14.3]])
    }

    >>> z = ivy.Container(a=ivy.zeros((2,2)), b=ivy.ones((2,2)))
    >>> x.random_normal(y, device='cpu', dtype='float64', out=z)
    {
        a: ivy.array([[10.8, 21.2],
                      [7.75, 4.65]]),
        b: ivy.array([[2.54, 5.01],
                      [11.7, 48.2]])
    }
    """
    return ivy.current_backend().random_normal(
        mean, std, shape, dtype=dtype, device=device, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@infer_device
@handle_nestable
def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Union[ivy.Array, ivy.NativeArray] = None,
    replace: bool = True,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.array:
    """
    Draws samples from a multinomial distribution. Specifically, returns a tensor
    where each row contains num_samples indices sampled from the multinomial probability
    distribution located in the corresponding row of tensor input.

    Parameters
    ----------
    population_size
        The size of the population from which to draw samples.
    num_samples
        Number of independent samples to draw from the population.
    batch_size
        Number of tensors to generate. Default is 1.
    probs
        The unnormalized probabilities for all elements in population,
        default is uniform *[batch_shape, population_size]*
    replace
        Whether to replace samples once they've been drawn. Default is True.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Drawn samples indices from the multinomial distribution.

    Examples
    --------
    >>> y = ivy.multinomial(10, 5)
    >>> print(y)
    ivy.array([[1, 8, 7, 8, 3]])

    >>> y = ivy.multinomial(10, 5, batch_size=2)
    >>> print(y)
    ivy.array([[9, 7, 9, 0, 7],
       [7, 3, 8, 5, 4]])

    >>> y = ivy.multinomial(10, 5, replace=False)
    >>> print(y)
    ivy.array([[2, 6, 4, 7, 0]])

    With :code:`ivy.Array` input:

    >>> y = ivy.multinomial(10, 5, probs=ivy.array([1/10]*10))
    >>> print(y)
    ivy.array([5, 2, 7, 6, 9])

    >>> y = ivy.multinomial(7, 5, batch_size=2, probs=ivy.array([[1/7]*7, [1/7]*7]))
    >>> print(y)
    ivy.array([[0, 4, 3, 4, 5], [1, 1, 0, 3, 2]])

    >>> y = ivy.multinomial(7, 5, batch_size=2, probs=ivy.array([[1/7]*7, [1/7]*7]),\
                            replace=False)
    >>> print(y)
    ivy.array([[2, 6, 1, 0, 3], [1, 0, 2, 5, 6]])

    With :code:`ivy.NativeArray` input:

    >>> y = ivy.multinomial(10, 5, probs=ivy.native_array([1/10]*10))
    >>> print(y)
    ivy.array([5, 7, 4, 2, 1])

    >>> y = ivy.multinomial(10, 5, batch_size=2,\
                            probs=ivy.native_array([[1/10]*10, [1/10]*10]))
    >>> print(y)
    ivy.array([[8, 0, 4, 1, 7], [2, 3, 4, 9, 3]])

    >>> y = ivy.multinomial(10, 5, batch_size=2,\
                     probs=ivy.native_array([[1/10]*10, [1/10]*10]), replace=False)
    >>> print(y)
    ivy.array([[0, 2, 6, 9, 1], [6, 7, 2, 4, 3]])

    """
    return ivy.current_backend().multinomial(
        population_size, num_samples, batch_size, probs, replace, device=device, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@infer_device
@handle_nestable
def randint(
    low: int,
    high: int,
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns an array filled with random integers generated uniformly between
    low (inclusive) and high (exclusive).

    Parameters
    ----------
    low
        Lowest integer that can be drawn from the distribution.
    high
        One above the highest integer that can be drawn from the distribution.
    shape
        a Sequence defining the shape of the output array.
    device
        device on which to create the array. 'cuda:0',
        'cuda:1', 'cpu' etc. (Default value = None).
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        Returns an array with the given shape filled with integers from
        the uniform distribution in the “half-open” interval [low, high)

    Examples
    --------
    >>> y = ivy.randint(0, 9, (1,1))
    >>> print(y)
    ivy.array([[5]])

    >>> y = ivy.randint(2, 20, (2, 2), device='cpu')
    >>> print(y)
    ivy.array([[5,8],[9,3]])

    >>> x = ivy.array([1, 2, 3])
    >>> ivy.randint(0, 10, (3,), out=x)
    >>> print(x)
    ivy.array([2, 6, 7])

    >>> y = ivy.zeros((3, 3))
    >>> ivy.randint(3, 15, (3, 3), device='cpu', out=y)
    >>> print(y)
    ivy.array([[ 7,  7,  5],
               [12,  8,  8],
               [ 8, 11,  3]])

    """
    return ivy.current_backend().randint(low, high, shape, device=device, out=out)


@handle_nestable
def seed(seed_value: int = 0) -> None:
    """Sets the seed for random number generation.

    Parameters
    ----------
    seed_value
        Seed for random number generation, must be a positive integer.
        (Default value = 0)

    Examples
    --------
    >>> ivy.seed(42)

    """
    return ivy.current_backend().seed(seed_value)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def shuffle(
    x: Union[ivy.Array, ivy.NativeArray], *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Shuffles the given array along axis 0.

    Parameters
    ----------
    x
        Input array. Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array object, shuffled along the first dimension.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3, 4, 5])
    >>> y = ivy.shuffle(x)
    >>> print(y)
    ivy.array([2, 1, 4, 3, 5])

    """
    return ivy.current_backend(x).shuffle(x, out=out)
