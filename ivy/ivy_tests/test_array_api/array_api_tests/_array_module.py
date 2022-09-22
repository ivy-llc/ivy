import os
from importlib import import_module

from . import stubs

# Replace this with a specific array module to test it, for example,
#
# import numpy as array_module
array_module = None

if array_module is None:
    if 'ARRAY_API_TESTS_MODULE' in os.environ:
        mod_name = os.environ['ARRAY_API_TESTS_MODULE']
        _module, _sub = mod_name, None
        if '.' in mod_name:
            _module, _sub = mod_name.split('.', 1)
        mod = import_module(_module)
        if _sub:
            try:
                mod = getattr(mod, _sub)
            except AttributeError:
                # _sub may be a submodule that needs to be imported. WE can't
                # do this in every case because some array modules are not
                # submodules that can be imported (like mxnet.nd).
                mod = import_module(mod_name)
    else:
        raise RuntimeError("No array module specified. Either edit _array_module.py or set the ARRAY_API_TESTS_MODULE environment variable")
else:
    mod = array_module
    mod_name = mod.__name__
# Names from the spec. This is what should actually be imported from this
# file.

class _UndefinedStub:
    """
    Standing for undefined names, so the tests can be imported even if they
    fail

    If this object appears in a test failure, it means a name is not defined
    in a function. This typically happens for things like dtype literals not
    being defined.

    """
    def __init__(self, name):
        self.name = name

    def _raise(self, *args, **kwargs):
        raise AssertionError(f"{self.name} is not defined in {mod_name}")

    def __repr__(self):
        return f"<undefined stub for {self.name!r}>"

    __call__ = _raise
    __getattr__ = _raise

_dtypes = [
    "bool",
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float32", "float64",
]
_constants = ["e", "inf", "nan", "pi"]
_funcs = [f.__name__ for funcs in stubs.category_to_funcs.values() for f in funcs]
_top_level_attrs = _dtypes + _constants + _funcs + stubs.EXTENSIONS

for attr in _top_level_attrs:
    try:
        globals()[attr] = getattr(mod, attr)
    except AttributeError:
        globals()[attr] = _UndefinedStub(attr)
