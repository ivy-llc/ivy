"""
Collection of MXNet random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

# local
from ivy.core.device import default_device
# noinspection PyProtectedMember
from ivy.mxnet.core.general import _mxnet_init_context
# noinspection PyProtectedMember
from ivy.mxnet.core.general import _1_dim_array_to_flat_array


def random_uniform(low=0., high=1., shape=None, dev_str=None):
    if isinstance(low, _mx.nd.NDArray):
        low = low.asscalar()
    if isinstance(high, _mx.nd.NDArray):
        high = high.asscalar()
    ctx = _mxnet_init_context(default_device(dev_str))
    if shape is None or len(shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.random.uniform(low, high, (1,), ctx=ctx))
    return _mx.nd.random.uniform(low, high, shape, ctx=ctx)


def random_normal(mean=0., std=1., shape=None, dev_str=None):
    if isinstance(mean, _mx.nd.NDArray):
        mean = mean.asscalar()
    if isinstance(std, _mx.nd.NDArray):
        std = std.asscalar()
    ctx = _mxnet_init_context(default_device(dev_str))
    if shape is None or len(shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.random.normal(mean, std, (1,), ctx=ctx))
    return _mx.nd.random.uniform(mean, std, shape, ctx=ctx)


def multinomial(population_size, num_samples, batch_size, probs=None, replace=True, dev_str=None):
    if not replace:
        raise Exception('MXNet does not support multinomial without replacement')
    ctx = _mxnet_init_context(default_device(dev_str))
    if probs is None:
        probs = _mx.nd.ones((batch_size, population_size,), ctx=ctx) / population_size
    probs = probs / _mx.nd.sum(probs, -1, True)
    return _mx.nd.sample_multinomial(probs, (num_samples,))


def randint(low, high, shape, dev_str=None):
    if isinstance(low, _mx.nd.NDArray):
        low = int(low.asscalar())
    if isinstance(high, _mx.nd.NDArray):
        high = int(high.asscalar())
    ctx = _mxnet_init_context(default_device(dev_str))
    if len(shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.random.randint(
            low, high, (1,), ctx=ctx))
    return _mx.nd.random.randint(low, high, shape, ctx=ctx)


seed = lambda seed_value=0: _mx.random.seed(seed_value)
shuffle = lambda x: _mx.nd.random.shuffle(x)
