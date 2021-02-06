"""
Collection of MXNet random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

# noinspection PyProtectedMember
from ivy.mxnd.core.general import _mxnet_init_context


def random_uniform(low, high, size, dev='cpu'):
    ctx = _mxnet_init_context(dev)
    return _mx.nd.random.uniform(low, high, size, ctx=ctx)


def randint(low, high, size, dev='cpu'):
    ctx = _mxnet_init_context(dev)
    return _mx.nd.random.randint(low, high, size, ctx=ctx)


seed = lambda seed_value: _mx.random.seed(seed_value)
shuffle = lambda x: _mx.nd.random.shuffle(x)
