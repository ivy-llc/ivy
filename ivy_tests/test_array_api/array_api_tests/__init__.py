from functools import wraps

from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

from ._array_module import mod as _xp

__all__ = ["xps"]

xps = make_strategies_namespace(_xp)


# We monkey patch floats() to always disable subnormals as they are out-of-scope

_floats = st.floats


@wraps(_floats)
def floats(*a, **kw):
    kw["allow_subnormal"] = False
    return _floats(*a, **kw)


st.floats = floats

from . import _version
__version__ = _version.get_versions()['version']
