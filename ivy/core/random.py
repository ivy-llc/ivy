"""
Collection of random Ivy functions
"""

# local
from ivy.framework_handler import get_framework as _get_framework


def random_uniform(low=0.0, high=1.0, shape=None, dev_str='cpu', f=None):
    """
    Draws samples from a uniform distribution.
    Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high).
    In other words, any value within the given interval is equally likely to be drawn by uniform.

    :param low: Lower boundary of the output interval. All values generated will be greater than or equal to low.
                The default value is 0.
    :type low: float
    :param high: Upper boundary of the output interval. All values generated will be less than high.
                The default value is 1.0.
    :type high: float
    :param shape: Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
                    If size is None (default), a single value is returned.
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Drawn samples from the parameterized uniform distribution.
    """
    return _get_framework(f=f).random_uniform(low, high, shape, dev_str)


def multinomial(probs, num_samples, dev_str='cpu', f=None):
    """
    Draws samples from a multinomial distribution.
    Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high).
    In other words, any value within the given interval is equally likely to be drawn by uniform.

    :param probs: The unnormalized log-probabilities for all classes *[batch_shape, num_classes]*
    :type probs: array
    :param num_samples: Number of independent samples to draw for each row slice
    :type num_samples: int
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Drawn samples from the parameterized uniform distribution.
    """
    return _get_framework(f=f).multinomial(probs, num_samples, dev_str)


def randint(low, high, shape, dev_str='cpu', f=None):
    """
    Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

    :param low: Lowest integer to be drawn from the distribution.
    :type low: int
    :param high: One above the highest integer to be drawn from the distribution.
    :type high: int
    :param shape: a tuple defining the shape of the output tensor.
    :type shape: tuple
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return:
    """
    return _get_framework(f=f).randint(low, high, shape, dev_str)


def seed(seed_value=0, f=None):
    """
    Sets the seed for random number generation.

    :param seed_value: Seed for random number generation, must be a positive integer.
    :type seed_value: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    """
    return _get_framework(f=f).seed(seed_value)


def shuffle(x, f=None):
    """
    Shuffles the given array along axis 0.

    :param x: An array object, in the specific Machine learning framework.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array object, shuffled along the first dimension.
    """
    return _get_framework(x, f=f).shuffle(x)
