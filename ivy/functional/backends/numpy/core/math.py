"""
Collection of Numpy math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np
try:
    from scipy.special import erf as _erf
except (ImportError, ModuleNotFoundError):
    _erf = None

tan = _np.tan
asin = _np.arcsin
acos = _np.arccos
atan = _np.arctan
atan2 = _np.arctan2
cosh = _np.cosh
atanh = _np.arctanh
log = _np.log
exp = _np.exp


def erf(x):
    if _erf is None:
        raise Exception('scipy must be installed in order to call ivy.erf with a numpy backend.')
    return _erf(x)
