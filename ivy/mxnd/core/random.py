"""
Collection of MXNet random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

# local
# noinspection PyProtectedMember
from ivy.mxnd.core.general import _mxnet_init_context
# noinspection PyProtectedMember
from ivy.mxnd.core.general import _1_dim_array_to_flat_array


def random_uniform(low=0., high=1., shape=None, dev_str='cpu'):
    if isinstance(low, _mx.nd.NDArray):
        low = low.asscalar()
    if isinstance(high, _mx.nd.NDArray):
        high = high.asscalar()
    ctx = _mxnet_init_context(dev_str)
    if shape is None or len(shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.random.uniform(low, high, (1,), ctx=ctx))
    return _mx.nd.random.uniform(low, high, shape, ctx=ctx)


def multinomial(population_size, num_samples, probs=None, replace=True):
    if not replace:
        raise Exception('MXNet does not support multinomial without replacement')
    if probs is None:
        probs = _mx.nd.ones((1, population_size,)) / population_size
    probs = probs / _mx.nd.sum(probs, -1, True)
    return _mx.nd.sample_multinomial(probs, (num_samples,))


def randint(low, high, shape, dev_str='cpu'):
    if isinstance(low, _mx.nd.NDArray):
        low = int(low.asscalar())
    if isinstance(high, _mx.nd.NDArray):
        high = int(high.asscalar())
    ctx = _mxnet_init_context(dev_str)
    if len(shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.random.randint(
            low, high, (1,), ctx=ctx))
    return _mx.nd.random.randint(low, high, shape, ctx=ctx)


seed = lambda seed_value=0: _mx.random.seed(seed_value)
shuffle = lambda x: _mx.nd.random.shuffle(x)
