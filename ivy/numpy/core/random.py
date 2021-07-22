"""
Collection of Numpy random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np

random_uniform = lambda low=0., high=1., shape=None, dev_str='cpu': _np.asarray(_np.random.uniform(low, high, shape))


def multinomial(population_size, num_samples, probs=None, replace=True):
    if probs is None:
        probs = _np.ones((1, population_size,)) / population_size
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = _np.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / _np.sum(probs_flat, -1, keepdims=True)
    probs_stack = _np.split(probs_flat, probs_flat.shape[0])
    samples_stack = [_np.random.choice(num_classes, num_samples, replace, p=prob[0]) for prob in probs_stack]
    samples_flat = _np.stack(samples_stack)
    return _np.asarray(_np.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples]))


randint = lambda low, high, shape, dev_str='cpu': _np.random.randint(low, high, shape)
seed = lambda seed_value=0: _np.random.seed(seed_value)


def shuffle(x):
    _np.random.shuffle(x)
    return x
