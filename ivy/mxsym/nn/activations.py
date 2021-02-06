"""
Collection of MXNet activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

relu = _mx.symbol.relu
leaky_relu = lambda x, alpha=0.2: _mx.symbol.LeakyReLU(x, slope=alpha)
tanh = _mx.symbol.tanh
sigmoid = _mx.symbol.sigmoid
softmax = _mx.symbol.softmax
softplus = lambda x: _mx.symbol.log(_mx.symbol.exp(x) + 1)
