"""Collection of random Ivy functions."""

# global
from typing import Optional, Union, Tuple

# local
from ivy.framework_handler import current_framework as _cur_framework
import ivy


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    device: Optional[ivy.Device] = None,
) -> ivy.array:
    """Draws samples from a uniform distribution. Samples are uniformly
    distributed over the half-open interval ``[low, high)`` (includes ``low``,
    but excludes ``high``). In other words, any value within the given interval
    is equally likely to be drawn by uniform.

    Parameters
    ----------
    low
        Lower boundary of the output interval. All values generated will be greater than or equal to ``low``.
    high
        Upper boundary of the output interval. All the values generated will be less than ``high``.
    shape
        If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples are drawn. If size is ``None``
        (Default), a single value is returned.
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
    return _cur_framework().random_uniform(low, high, shape, device)


def random_normal(mean=0.0, std=1.0, shape=None, device=None):
    """Draws samples from a normal distribution.

    Parameters
    ----------
    mean
        The mean of the normal distribution to sample from. Default is 0.
    std
        The standard deviation of the normal distribution to sample from. Default is 1.
    shape
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        If size is None (default), a single value is returned.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)

    Returns
    -------
    ret
        Drawn samples from the parameterized uniform distribution.

    """
    return _cur_framework().random_normal(mean, std, shape, device)


def multinomial(
    population_size, num_samples, batch_size, probs=None, replace=True, device=None
):
    """Draws samples from a multinomial distribution. Specifcally, returns a
    tensor where each row contains num_samples indices sampled from the
    multinomial probability distribution located in the corresponding row of
    tensor input.

    Parameters
    ----------
    population_size
        The size of the population from which to draw samples.
    num_samples
        Number of independent samples to draw from the population.
    batch_size
        Number of times to draw a new set of samples from the population.
    probs
        The unnormalized probabilities for all elemtns in population,
        default is uniform *[batch_shape, num_classes]*
    replace
        Whether to replace samples once they've been drawn. Default is True.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)

    Returns
    -------
    ret
        Drawn samples indices from the multinomial distribution.

    """
    return _cur_framework().multinomial(
        population_size, num_samples, batch_size, probs, replace, device
    )


def randint(low, high, shape, device=None):
    """Returns a tensor filled with random integers generated uniformly between
    low (inclusive) and high (exclusive).

    Parameters
    ----------
    low
        Lowest integer to be drawn from the distribution.
    high
        One above the highest integer to be drawn from the distribution.
    shape
        a tuple defining the shape of the output tensor.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)

    """
    return _cur_framework().randint(low, high, shape, device)


def seed(seed_value=0):
    """Sets the seed for random number generation.

    Parameters
    ----------
    seed_value
        Seed for random number generation, must be a positive integer. (Default value = 0)

    """
    return _cur_framework().seed(seed_value)


def shuffle(x):
    """Shuffles the given array along axis 0.

    Parameters
    ----------
    x
        An array object, in the specific Machine learning framework.

    Returns
    -------
    ret
        An array object, shuffled along the first dimension.

    """
    return _cur_framework(x).shuffle(x)
