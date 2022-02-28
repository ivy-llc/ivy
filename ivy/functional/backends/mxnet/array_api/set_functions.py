import mxnet as mx
from typing import Tuple
from collections import namedtuple

def unique_counts(x: mx.array) \
                -> Tuple[mx.array, mx.array]:
    uc = namedtuple('uc', ['values', 'counts'])
    v, c = mx.unique(x, return_counts=True)
    return uc(v, c)