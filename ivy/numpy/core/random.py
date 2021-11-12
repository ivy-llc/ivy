"""
Collection of Numpy random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np

random_uniform = lambda low=0., high=1., shape=None, dev_str=None: _np.asarray(_np.random.uniform(low, high, shape))
random_normal = lambda mean=0., std=1., shape=None, dev_str=None: _np.asarray(_np.random.normal(mean, std, shape))


def multinomial(population_size, num_samples, batch_size, probs=None, replace=True, dev_str=None):
    if probs is None:
        probs = _np.ones((batch_size, population_size,)) / population_size
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = _np.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / _np.sum(probs_flat, -1, keepdims=True)
    probs_stack = _np.split(probs_flat, probs_flat.shape[0])
    samples_stack = [_np.random.choice(num_classes, num_samples, replace, p=prob[0]) for prob in probs_stack]
    samples_flat = _np.stack(samples_stack)
    return _np.asarray(_np.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples]))


randint = lambda low, high, shape, dev_str=None: _np.random.randint(low, high, shape)
seed = lambda seed_value=0: _np.random.seed(seed_value)
shuffle = _np.random.permutation
