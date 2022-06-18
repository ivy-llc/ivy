"""Collection of random Ivy functions."""

# global
from typing import Optional, Union, Tuple, Sequence

# local
from ivy.backend_handler import current_backend as _cur_backend
from ivy.func_wrapper import (
    infer_device,
    outputs_to_ivy_arrays,
    handle_out_argument,
    to_native_arrays_and_back,
)
import ivy


# Extra #
# ------#


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    dtype=None,
) -> ivy.array:
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

    Returns
    -------
    ret
        Drawn samples from the parameterized uniform distribution.

    Examples
    --------
    >>> y = ivy.random_uniform(0.0, 2.0)
    >>> print(y)
    ivy.array(1.89150229)

    """
    dtype = ivy.default_dtype(dtype, as_native=True)
    return _cur_backend().random_uniform(low, high, shape, device=device, dtype=dtype)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
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

    Returns
    -------
     ret
        Drawn samples from the parameterized normal distribution.

    Examples
    --------
    >>> y = ivy.random_normal(0.0, 2.0)
    >>> print(y)
    ivy.array(0.6444774682897879)
    """
    return _cur_backend().random_normal(mean, std, shape, device=device)


@to_native_arrays_and_back
@handle_out_argument
@infer_device
def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Union[ivy.Array, ivy.NativeArray] = None,
    replace: bool = True,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
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
    return _cur_backend().multinomial(
        population_size, num_samples, batch_size, probs, replace, device=device
    )


@to_native_arrays_and_back
@handle_out_argument
@infer_device
def randint(
    low: int,
    high: int,
    shape: Union[int, Sequence[int]],
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
    >>> y = ivy.randint(0, 9, 1)
    >>> print(y)
    ivy.array([3])

    >>> y = ivy.randint(2, 20, (2, 2), 'cpu')
    >>> print(y)
    ivy.array([[ 7,  5],
               [15, 15]])

    >>> x = ivy.Array([1, 2, 3])
    >>> ivy.randint(0, 10, (3,), out=x)
    >>> print(x)
    ivy.array([2, 6, 7])

    >>> y = ivy.zeros(3, 3)
    >>> ivy.randint(3, 15, (3, 3), 'gpu:1', out=y)
    >>> print(y)
    ivy.array([[ 7,  7,  5],
               [12,  8,  8],
               [ 8, 11,  3]])

    """
    return _cur_backend().randint(low, high, shape, device=device)


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
    return _cur_backend().seed(seed_value)


@to_native_arrays_and_back
@handle_out_argument
def shuffle(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """Shuffles the given array along axis 0.

    Parameters
    ----------
    x
        Input array. Should have a numeric data type.

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
    return _cur_backend(x).shuffle(x)
