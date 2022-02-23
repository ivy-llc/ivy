"""
Collection of Numpy math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np
try:
    from scipy.special import erf as _erf
except (ImportError, ModuleNotFoundError):
    _erf = None

sin = _np.sin
cos = _np.cos
tan = _np.tan
asin = _np.arcsin
acos = _np.arccos
atan = _np.arctan
atan2 = _np.arctan2
sinh = _np.sinh
cosh = _np.cosh
tanh = _np.tanh
asinh = _np.arcsinh
acosh = _np.arccosh
atanh = _np.arctanh
log = _np.log
exp = _np.exp


def erf(x):
    if _erf is None:
        raise Exception('scipy must be installed in order to call ivy.erf with a numpy backend.')
    return _erf(x)
