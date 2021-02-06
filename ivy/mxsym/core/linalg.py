"""
Collection of MXNet linear algebra functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

# local


def svd(*_):
    raise Exception('ivy.linalg.svd() not supported in mxnet symbolic mode.')


# noinspection PyShadowingBuiltins
norm = lambda x, ord=2, axis=-1, keepdims=False: _mx.symbol.norm(x, ord=ord, axis=axis, keepdims=keepdims)
inv = _mx.symbol.linalg_inverse


def pinv(*_):
    raise Exception('MXNet does not support pinv().')


def vector_to_skew_symmetric_matrix(*_):
    raise Exception('mxnet symbolic does not support ivy.linalg.vector_to_skew_symmetric_matrix(),'
                    'as symbolic objects are not fully sliceable')
