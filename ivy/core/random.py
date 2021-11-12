"""
Collection of random Ivy functions
"""

# local
from ivy.framework_handler import current_framework as _cur_framework


def random_uniform(low=0.0, high=1.0, shape=None, dev_str=None, f=None):
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
    :type shape: sequence of ints
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Drawn samples from the parameterized uniform distribution.
    """
    return _cur_framework(f=f).random_uniform(low, high, shape, dev_str)


def random_normal(mean=0.0, std=1.0, shape=None, dev_str=None, f=None):
    """
    Draws samples from a normal distribution.

    :param mean: The mean of the normal distribution to sample from. Default is 0.
    :type mean: float
    :param std: The standard deviation of the normal distribution to sample from. Default is 1.
    :type std: float
    :param shape: Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
                    If size is None (default), a single value is returned.
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Drawn samples from the parameterized uniform distribution.
    """
    return _cur_framework(f=f).random_normal(mean, std, shape, dev_str)


def multinomial(population_size, num_samples, batch_size, probs=None, replace=True, dev_str=None, f=None):
    """
    Draws samples from a multinomial distribution. Specifcally, returns a tensor where each row contains num_samples
    indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.

    :param population_size: The size of the population from which to draw samples.
    :type population_size: int
    :param num_samples: Number of independent samples to draw from the population.
    :type num_samples: int
    :param batch_size: Number of times to draw a new set of samples from the population.
    :type num_samples: int
    :param probs: The unnormalized probabilities for all elemtns in population,
                        default is uniform *[batch_shape, num_classes]*
    :type probs: array, optional
    :param replace: Whether to replace samples once they've been drawn. Default is True.
    :type replace: bool, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Drawn samples indices from the multinomial distribution.
    """
    return _cur_framework(f=f).multinomial(population_size, num_samples, batch_size, probs, replace, dev_str)


def randint(low, high, shape, dev_str=None, f=None):
    """
    Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

    :param low: Lowest integer to be drawn from the distribution.
    :type low: int
    :param high: One above the highest integer to be drawn from the distribution.
    :type high: int
    :param shape: a tuple defining the shape of the output tensor.
    :type shape: sequence of ints
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return:
    """
    return _cur_framework(f=f).randint(low, high, shape, dev_str)


def seed(seed_value=0, f=None):
    """
    Sets the seed for random number generation.

    :param seed_value: Seed for random number generation, must be a positive integer.
    :type seed_value: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    """
    return _cur_framework(f=f).seed(seed_value)


def shuffle(x, f=None):
    """
    Shuffles the given array along axis 0.

    :param x: An array object, in the specific Machine learning framework.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array object, shuffled along the first dimension.
    """
    return _cur_framework(x, f=f).shuffle(x)
