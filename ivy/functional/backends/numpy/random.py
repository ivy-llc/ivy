"""
Collection of Numpy random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as np
from typing import Optional, Union, Tuple

# local
import ivy

# Extra #
# ------#

def random_uniform(low: float = 0.0,
                   high: float = 1.0,
                   shape: Optional[Union[int, Tuple[int, ...]]] = None,
                   device: Optional[ivy.Device] = None) -> np.ndarray:
    return np.asarray(np.random.uniform(low, high, shape))


random_normal = lambda mean=0., std=1., shape=None, device=None: np.asarray(np.random.normal(mean, std, shape))


def multinomial(population_size, num_samples, batch_size, probs=None, replace=True, device=None):
    if probs is None:
        probs = np.ones((batch_size, population_size,)) / population_size
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = np.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / np.sum(probs_flat, -1, keepdims=True)
    probs_stack = np.split(probs_flat, probs_flat.shape[0])
    samples_stack = [np.random.choice(num_classes, num_samples, replace, p=prob[0]) for prob in probs_stack]
    samples_flat = np.stack(samples_stack)
    return np.asarray(np.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples]))


randint = lambda low, high, shape, device=None: np.random.randint(low, high, shape)
seed = lambda seed_value=0: np.random.seed(seed_value)


def shuffle(x):
    return np.random.permutation(x)
