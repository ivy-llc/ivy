"""
Collection of random Ivy functions
"""

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
    """
    Draws samples from a uniform distribution. Samples are uniformly distributed over the half-open
    interval ``[low, high)`` (includes ``low``, but excludes ``high``). In other words, any value within the given
    interval is equally likely to be drawn by uniform.

    Parameters
    -----------
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


def random_normal(mean: float =0.0,
                  std: float = 1.0,
                  shape: Optional[Union[int, Tuple[int, ...]]] = None,
                  device: Optional[ivy.Device] = None) \
                  -> ivy.Array:
    """
    Draws samples from a normal distribution.

    Parameters
    ----------
    mean:
        The mean of the normal distribution to sample from. Default is 0.
    std:
        The standard deviation of the normal distribution to sample from. Default is 1.
    shape:
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        If size is None (default), a single value is returned.
    device:
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)

    Returns
    -------
    ret:
        Drawn samples from the parameterized uniform distribution.
        
    Example
    -------
    >>> x = ivy.random_normal(mean=0.0, std=1.0, shape=(4, 4))
    >>> print(x)
    ivy.array([[ 2.1141,  0.8101,  0.9298,  0.8460],
               [-1.2119, -0.3519, -0.6252,  0.4033],
               [ 0.7443,  0.2577, -0.3707, -0.0545],
               [-0.3238,  0.5944,  0.0775, -0.4327]])

    """
    return _cur_framework().random_normal(mean, std, shape, device)


def multinomial(population_size: int,
                num_samples: int,
                batch_size: int,
                probs: Optional[Union[ivy.Array, ivy.NativeArray]]=None,
                replace: bool=True,
                device: Optional[ivy.Device]=None)\
                -> ivy.Array:
    """
    Draws samples from a multinomial distribution. Specifcally, returns a tensor where each row contains num_samples
    indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.

    Parameters
    ----------
    population_size:
        The size of the population from which to draw samples.
    num_samples:
        Number of independent samples to draw from the population.
    batch_size:
        Number of times to draw a new set of samples from the population.
    probs:
        The unnormalized probabilities for all elemtns in population,
        default is uniform *[batch_shape, num_classes]*
    replace:
        Whether to replace samples once they've been drawn. Default is True.
    device:
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)

    Returns
    -------
    ret:
        Drawn samples indices from the multinomial distribution.
        
    Example
    -------
    >>> ret = ivy.multinomial(population_size = 5, num_samples = 4, batch_size = 3)
    >>> print(ret)
    ivy.array([[0, 1, 3, 4],
               [2, 3, 2, 0],
               [4, 4, 1, 0]])
    """
    return _cur_framework().multinomial(
        population_size, num_samples, batch_size, probs, replace, device
    )


def randint(low: int,
            high: int,
            shape: Union[int, Tuple[int, ...]],
            device: Optional[ivy.Device]=None) \
            -> ivy.Array:
    """
    Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

    Parameters
    ----------
    low:
        Lowest integer to be drawn from the distribution.
    high:
        One above the highest integer to be drawn from the distribution.
    shape:
        a tuple defining the shape of the output tensor.
    device:
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)

    Returns
    -------
    out:
        `size`-shaped array of random integers from the appropriate distribution, or a single such random int if `size` not provided.
        
    Example
    -------
    >>> x = ivy.randint(low = 0, high = 10, shape = (2, 10))
    >>> print(x)
    ivy.array([[4, 1, 6, 7, 5, 4, 0, 2, 7, 8],
               [4, 5, 9, 1, 1, 0, 5, 8, 9, 2]])
    """
    return _cur_framework().randint(low, high, shape, device)


def seed(seed_value=0):
    """Sets the seed for random number generation.

    Parameters
    ----------
    seed_value
        Seed for random number generation, must be a positive integer. (Default value = 0)

    Returns
    -------

    """
    return _cur_framework().seed(seed_value)


def shuffle(x: Union[ivy.Array, ivy.NativeArray]) \
            -> ivy.Array:
    """
    Shuffles the given array along axis 0.

    Parameters
    ----------
    x:
        An array object, in the specific Machine learning framework.

    Returns
    -------
    ret:
        An array object, shuffled along the first dimension.
        
    Example
    -------
    >>> x = ivy.random_normal(mean=0.0, std=1.0, shape=(4, 4))
    >>> print(x)
    ivy.array([[-1.3250, -0.2924,  0.9763, -0.6396],
               [-2.1592,  0.2214,  0.3466, -0.4488],
               [ 0.2806,  0.5338,  0.6362, -1.0206],
               [-2.3644,  1.5784,  2.9821,  0.2933]])
    >>> x = ivy.shuffle(x)
    >>> print(x)
    ivy.array([[ 0.2806,  0.5338,  0.6362, -1.0206],
               [-2.3644,  1.5784,  2.9821,  0.2933],
               [-1.3250, -0.2924,  0.9763, -0.6396],
               [-2.1592,  0.2214,  0.3466, -0.4488]])
    
    """
    return _cur_framework(x).shuffle(x)
