"""Collection of random Ivy functions."""

# global
from typing import Optional, Union

# local
import ivy
from ivy.func_wrapper import (
    infer_device,
    infer_dtype,
    outputs_to_ivy_arrays,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
)


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
    dtype=None,
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
        or equal to ``low``.
    high
        Upper boundary of the output interval. All the values generated will be less
        than ``high``.
    shape
        If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples are drawn.
        If size is ``None`` (Default), a single value is returned.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None).
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Drawn samples from the parameterized uniform distribution.

    Functional Examples
    -------------------

    >>> y = ivy.random_uniform()
    >>> print(y)
    ivy.array(0.26431865)

    >>> y = ivy.random_uniform(shape=3)
    >>> print(y)
    ivy.array([0.475, 0.878, 0.861])

    >>> y = ivy.random_uniform(0.0, 2.0, device="cpu")
    >>> print(y)
    ivy.array(1.89150229)

    >>> y = ivy.random_uniform(0.7, 1.0, device="cpu", shape=(2, 2))
    >>> print(y)
    ivy.array([[0.89629126, 0.94198485],
               [0.91405606, 0.72848724]])

    Instance Method Examples
    ------------------------

    With :code:`ivy.Container` input:

    >>> y = ivy.Container(a=ivy.random_uniform(), \
                          b=ivy.random_uniform(shape=2))
    >>> print(y)
    {
    a: ivy.array(0.7550739),
    b: ivy.array([0.624, 0.00109])
    }

    """
    return ivy.current_backend().random_uniform(
        low, high, shape, device=device, dtype=dtype, out=out
    )


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
@handle_nestable
def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.array:
    """
    Draws samples from a normal distribution.

    Parameters
    ----------
    mean
        The mean of the normal distribution to sample from. Default is ``0``.
    std
        The standard deviation of the normal distribution to sample from.
        Default is ``1``.
    shape
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k``
        samples are drawn. If size is ``None`` (default), a single value is returned.
    device
        (Default value = ``None``)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
     ret
        Drawn samples from the parameterized normal distribution.

    Funtional Examples
    ------------------

    >>> y = ivy.random_normal(0.0, 2.0)
    >>> print(y)
    ivy.array(0.6444774682897879)

    >>> y = ivy.random_normal(shape=3)
    >>> print(y)
    ivy.array([ 0.811, -0.508, -0.564])

    >>> y = ivy.random_normal(0.0,2.0,device='cpu')
    >>> print(y)
    ivy.array(-0.7268672)

    >>> y = ivy.random_normal(0.7, 1.0, device="cpu", shape=(2, 2))
    >>> print(y)
    ivy.array([[1.17 , 0.968],
               [0.175, 0.064]])

    Instance Method Examples
    ------------------------

    With :code:`ivy.Container` input:

    >>> y = ivy.Container(a=ivy.random_normal(), \
                          b=ivy.random_normal(shape=2))
    >>> print(y)
    {
    a: ivy.array(-0.40935726),
    b: ivy.array([1.54 , 0.556])
    }

    """
    return ivy.current_backend().random_normal(mean, std, shape, device=device, out=out)


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
        default is uniform *[batch_shape, num_classes]*
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


@outputs_to_ivy_arrays
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
