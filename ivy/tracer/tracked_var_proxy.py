from typing import Any
import importlib
import logging
import copy
import enum
import ivy
from . import globals as glob

# helpers

# ToDo: Docs
# ToDo: Find a dynamic alternative to method definition
# so that all of them mirror the corresponding .var method


"""Convenient functions"""


def get_types_to_ignore():
    """Method to get types to ignore when deciding to go
    deeper into the nest in the nest functions."""
    from_modules = []
    for _module, _class in [
        ("torch", "Size"),
        ("tensorflow", "TensorShape"),
    ]:
        if _module in [ivy.current_backend_str(), "ivy"]:
            try:
                _type = getattr(importlib.import_module(_module), _class)
                from_modules.append(_type)
            except:
                pass
    return tuple(from_modules + [TrackedVarProxy])


def _override_tensor_shape_proxy() -> bool:
    """Method to override the tracked tensor shape proxy
    if the backend is set to `tensorflow`."""
    _tf_tensor_shape_override = False

    try:
        global TrackedVarProxy, TrackedTensorShapeProxy
        global TYPE_TO_PROXY

        if ivy.current_backend_str() == "tensorflow":
            # Replace with a new tracked proxy class
            # with tf.TensorShape imported and inherited
            cls = TrackedTensorShapeProxyMeta(
                "TrackedTensorShapeProxy",
                TrackedTensorShapeProxy.__bases__,
                TrackedTensorShapeProxy.__dict__,
            )
            TYPE_TO_PROXY["TensorShape"] = cls
            _tf_tensor_shape_override = True

    except Exception as _:
        pass

    return _tf_tensor_shape_override


def proxy_classes() -> set:
    """Interface to retrieve a set of proxy classes."""
    global TYPE_TO_PROXY, PROXY_TYPE_TO_ITERATOR_PROXY_TYPE, _tf_tensor_shape_override

    # Check if we have overridden the TrackedTensorShape class.
    # If not, override it if necessary, before retreiving the dict
    if not _tf_tensor_shape_override:
        _tf_tensor_shape_override = _override_tensor_shape_proxy()

    return set([c for c in TYPE_TO_PROXY.values()] + [TrackedVarProxy]).union(
        set([c for c in PROXY_TYPE_TO_ITERATOR_PROXY_TYPE.values()])
    )


def type_to_proxy() -> dict:
    """Interface to retrieve a dictionary mapping of types to
    their tracked proxy classes."""
    global TYPE_TO_PROXY, _tf_tensor_shape_override

    # Check if we have overridden the TrackedTensorShape class.
    # If not, override it if necessary, before retreiving the dict
    if not _tf_tensor_shape_override:
        _tf_tensor_shape_override = _override_tensor_shape_proxy()

    return TYPE_TO_PROXY


def is_tracked_slice(s):
    """Method to check whether a slice contains Tracked vars inside."""
    return isinstance(s, slice) and any(
        [
            isinstance((getattr(s, p)), TrackedVarProxy)
            for p in ["start", "step", "stop"]
        ]
    )


def slice_to_list(s):
    """Converts a given slice to a list."""
    return [s.start, s.stop, s.step]


def list_to_slice(l):
    """Converts a given list to a slice."""
    return slice(*l)


def lazy_fn(fn_name):
    """Defers function getting to ensure that the
    function is wrapped when transpiling."""

    def lazy_tvp_placeholder(*args, **kwargs):
        return getattr(TrackedVarProxy, fn_name)(*args, **kwargs)

    return lazy_tvp_placeholder


def should_be_tracked(fn, att_name, ret, backend):
    """Checks whether the return value from a function
    with a given backend should be tracked or not."""
    fn_name = fn.__name__
    fn_module = fn.__module__ if hasattr(fn, "__module__") else None
    if type(ret).__name__ in glob.CLASSES_TO_TRACK[backend]:
        return True
    if fn_name in glob.FNS_TO_TRACK[backend] or fn_name in glob.BUILTIN_FNS_TO_TRACK:
        return True
    if fn_module in glob.BUILTIN_MODULES_TO_TRACK:
        return True
    if fn_name in ["__getattr__", "__getattribute__"]:
        return att_name in glob.ATTRS_TO_TRACK[backend]
    return fn_name == "wrapped_builtin_callable"


def should_not_be_logged(fn_name, args, att_name, backend):
    """Convenient function used to prevent logging of some attrs
    like `jax.shape` when called internally via some private attr
    that we are already not logging."""
    if fn_name not in ["__getattr__", "__getattribute__"]:
        return False
    for arg in args:
        if type(arg).__name__ in glob.CLASS_ATTRS_NOT_TO_TRACK[backend]:
            return (
                att_name in glob.CLASS_ATTRS_NOT_TO_TRACK[backend][type(arg).__name__]
            )
    return False


def log_warning(fn):
    if hasattr(fn, "__name__"):
        w_msg = f"Usage of {fn.__name__} on a TrackedVar breaks tracking."
        if f"tvp{fn.__name__}" in dir(TrackedVarProxy):
            fn_name = fn.__name__.replace("_", "")
            w_msg += (
                f" If this function is being called as {fn_name}(), you can replace it "
                + f"with tracer.{fn_name}()\n Check the docs for more info: <docs>"
            )
        logging.warning(w_msg)


def unset_tf_tensor_shape_override():
    """Convenient function to reset the behaviour of `TrackedTensorShapeProxy`
    class such that it doesn't inherit from `tf.TensorShape` anymore."""
    global _tf_tensor_shape_override, TYPE_TO_PROXY
    global TrackedTensorShapeProxy

    _tf_tensor_shape_override = False
    TYPE_TO_PROXY["TensorShape"] = TrackedTensorShapeProxy


"""Classes"""


# Base Class
class TrackedVarProxy:
    """A class to wrap variables that should be tracked during compilation.
    When wrapping a variable, the corresponding dtype attributes are available
    through this class.


    Attributes
    ----------
    var
        the variable that is being wrapped."""

    is_tracked_proxy = True

    def __init__(self, a):
        self.var = a

    def get_var(self):
        return self.var

    # this function is called in ivy
    def override_dtype_check(self):
        return ivy.default_dtype(item=self.get_var())

    def __copy__(self):
        cls = self.__class__
        result = cls(self.get_var())
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls(copy.deepcopy(self.get_var()))
        return result

    # static methods

    @staticmethod
    def tvp__len__(x):
        return len(x)

    @staticmethod
    def tvp__int__(x):
        return int(x)

    @staticmethod
    def tvp__float__(x):
        return float(x)


# TrackedIntProxy
class TrackedIntProxy(int, TrackedVarProxy):
    def __new__(cls, v):
        return super(TrackedIntProxy, cls).__new__(cls, int(v))


# TrackedFloatProxy
class TrackedFloatProxy(float, TrackedVarProxy):
    def __new__(cls, v):
        return super(TrackedFloatProxy, cls).__new__(cls, float(v))


# TrackedStrProxy
class TrackedStrProxy(str, TrackedVarProxy):
    def __new__(cls, v):
        return super(TrackedStrProxy, cls).__new__(cls, str(v))

    def __init__(self, v):
        super().__init__(str(v))
        self.var = v

    def __iter__(self):
        return TrackedStrIteratorProxy(self)

    def __len__(self):
        return len(self.var)


# TrackedListProxy
class TrackedListProxy(list, TrackedVarProxy):
    def __new__(cls, v):
        return super(TrackedListProxy, cls).__new__(cls, list(v))

    def __init__(self, v):
        super().__init__(list(v))
        self.var = v

    def __iter__(self):
        return TrackedListIteratorProxy(self)

    def __len__(self):
        return len(self.var)


# TrackedTupleProxy
class TrackedTupleProxy(tuple, TrackedVarProxy):
    def __new__(cls, v):
        return super(TrackedTupleProxy, cls).__new__(cls, tuple(v))

    def __init__(self, v):
        super().__init__(tuple(v))
        self.var = tuple(v)

    def __iter__(self):
        return TrackedTupleIteratorProxy(self)

    def __len__(self):
        return len(self.var)

    def numel(self):
        import torch

        return torch.Size(self).numel()


# TrackedDictProxy
class TrackedDictProxy(dict, TrackedVarProxy):
    def __new__(cls, v):
        return super(TrackedDictProxy, cls).__new__(cls, dict(v))

    def __init__(self, v):
        super().__init__(dict(v))
        self.var = v

    def __iter__(self):
        return TrackedDictKeysIteratorProxy(self)

    def __len__(self):
        return len(self.var)

    def __getitem__(self, key):
        return self.var.__getitem__(key)

    def __setitem__(self, key, value):
        self.var.__setitem__(key, value)

    def __delitem__(self, key):
        self.var.__delitem__(key)

    def keys(self):
        return TrackedDictKeysIteratorProxy(self.var)

    def values(self):
        return TrackedDictValuesIteratorProxy(self.var)

    def items(self):
        return TrackedDictItemsIteratorProxy(self.var)


# TrackedBytesProxy
class TrackedBytesProxy(bytes, TrackedVarProxy):
    def __new__(cls, v):
        return super(TrackedBytesProxy, cls).__new__(cls, bytes(v))

    def __init__(self, v):
        super().__init__(bytes(v))
        self.var = v

    def __iter__(self):
        return TrackedBytesIteratorProxy(self)

    def __len__(self):
        return len(self.var)


# TrackedEnumProxy
class TrackedEnumProxy(TrackedVarProxy, enum.Enum):
    @classmethod
    def _missing_(cls, candidate):
        candidate_cls = candidate.__class__
        for member in candidate_cls:
            if member.value == candidate.value:
                # Create a new object
                obj = object.__new__(cls)

                # Copy over the attributes
                obj._name_ = member.name
                obj._value_ = member.value

                # Initialize super to get access to get_var instance methods
                cls.__init__(obj, member)

                # Copy over the metadata
                obj._value2member_map_ = candidate_cls._value2member_map_
                obj._member_map_ = candidate_cls._member_map_
                obj._member_names_ = candidate_cls._member_names_

                return obj

    def __init__(self, v):
        super().__init__(v)
        self.var = v

    def __getattribute__(self, __name: str) -> Any:
        if __name in ["_value2member_map_", "_member_map_", "_member_names_"]:
            return self.__getattr__(__name)
        return super(enum.Enum, self).__getattribute__(__name)

    def __getattr__(self, attr):
        try:
            return getattr(self.get_var(), attr)
        except AttributeError as e:
            raise e


# TrackedIntEnumProxy
class TrackedIntEnumProxy(TrackedVarProxy, enum.IntEnum):
    @classmethod
    def _missing_(cls, candidate):
        candidate_cls = candidate.__class__
        for member in candidate_cls:
            if member.value == candidate.value:
                # Create a new object
                obj = int.__new__(cls, member.value)
                super(cls, obj).__init__(obj)

                # Copy over the attributes
                obj._name_ = member.name
                obj._value_ = member.value

                # Initialize super to get access to get_var instance methods
                cls.__init__(obj, member)

                # Copy over the metadata
                obj._value2member_map_ = candidate_cls._value2member_map_
                obj._member_map_ = candidate_cls._member_map_
                obj._member_names_ = candidate_cls._member_names_

                return obj

    def __init__(self, v):
        super().__init__(v)
        self.var = v

    def __getattribute__(self, __name: str) -> Any:
        if __name in ["_value2member_map_", "_member_map_", "_member_names_"]:
            return self.__getattr__(__name)
        return super(enum.IntEnum, self).__getattribute__(__name)

    def __getattr__(self, attr):
        try:
            return getattr(self.get_var(), attr)
        except AttributeError as e:
            raise e


# TrackedShapeProxy
class TrackedShapeProxy(ivy.Shape, TrackedVarProxy):
    def __new__(cls, v):
        return super(TrackedShapeProxy, cls).__new__(cls)

    def __init__(self, v):
        super().__init__(v)
        self.var = v

    def __iter__(self):
        return TrackedShapeIteratorProxy(self)

    def __len__(self):
        return len(self.var)


# TrackedTensorShapeProxy
# This class is overridden using the metaclass based on
# whether we are compiling to tensorflow or not. This avoids
# having to import tensorflow unnecessarily when it is not required.
class TrackedTensorShapeProxy(TrackedVarProxy):
    def __iter__(self):
        return TrackedTupleIteratorProxy(self)

    def __repr__(self):
        return repr(self.var)


"""Metaclasses"""


# Metaclass for TrackedTensorShapeProxy
class TrackedTensorShapeProxyMeta(type):
    def __new__(cls, name, bases, clsdict, **kwargs):
        def new__init__(self, v):
            tf.TensorShape.__init__(self, dims=v)
            TrackedVarProxy.__init__(self, a=v)
            self.var = v

        new__init__.__name__ = "__init__"
        new__init__.__qualname__ = f"{name}.__init__"

        import tensorflow as tf

        bases = (tf.TensorShape,) + bases
        new_cls = type(name, bases, dict(clsdict))
        new_cls.__init__ = new__init__

        return new_cls


"""Iterator Classes"""


# Iterator Class for TrackedStrProxy
class TrackedStrIteratorProxy(TrackedVarProxy):
    def __init__(self, v):
        self.var = v
        self._index = 0

    def __next__(self):
        if self._index >= len(self.var):
            raise StopIteration
        value = self.var[self._index]
        self._index += 1
        return value

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.var)

    def __reversed__(self):
        return TrackedStrIteratorProxy(reversed(self.var))

    def __contains__(self, item):
        return item in self.var

    def __repr__(self):
        return f"TrackedStrIteratorProxy({self.var})"


# Iterator Class for TrackedListProxy
class TrackedListIteratorProxy(TrackedVarProxy):
    def __init__(self, v):
        self.var = v
        self._index = 0

    def __next__(self):
        if self._index >= len(self.var):
            raise StopIteration
        value = self.var[self._index]
        self._index += 1
        return value

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.var)

    def __reversed__(self):
        return TrackedListIteratorProxy(reversed(self.var))

    def __contains__(self, item):
        return item in self.var

    def __repr__(self):
        return f"TrackedListIteratorProxy({self.var})"


# Iterator Class for TrackedTupleProxy
class TrackedTupleIteratorProxy(TrackedVarProxy):
    def __init__(self, v):
        self.var = v
        self._index = 0

    def __next__(self):
        if self._index >= len(self.var):
            raise StopIteration
        value = self.var[self._index]
        self._index += 1
        return value

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.var)

    def __getitem__(self, index):
        return self.var[index]

    def __reversed__(self):
        return TrackedTupleIteratorProxy(reversed(self.var))

    def __contains__(self, item):
        return item in self.var

    def __repr__(self):
        return f"TrackedTupleIteratorProxy({self.var})"


# Iterator class for TrackedShapeProxy
class TrackedShapeIteratorProxy(TrackedVarProxy):
    def __init__(self, v):
        self.var = v
        self._index = 0
        self._iterator = None

    def __next__(self):
        if self._index == 0 and self._iterator is None:
            if isinstance(self.var.shape, tuple):
                self._iterator = TrackedTupleIteratorProxy(self.var.shape)
            elif isinstance(self.var.shape, list):
                self._iterator = TrackedListIteratorProxy(self.var.shape)
            else:
                self._iterator = iter(self.var.shape)

        value = self._iterator.__next__()
        return value

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.var)

    def __getitem__(self, index):
        return self.var[index]

    def __reversed__(self):
        return TrackedShapeIteratorProxy(reversed(self.var))

    def __contains__(self, item):
        return item in self.var

    def __repr__(self):
        return f"TrackedShapeIteratorProxy({self.var})"


# Iterator Class for TrackedDictProxy
class TrackedDictIteratorProxy(TrackedVarProxy):
    def __init__(self, v, mode: str = "items"):
        self.var = v
        self._mode = mode
        self._keys = list(self.var.keys())
        self._values = list(self.var.values())
        self._items = list(self.var.items())
        self._index = 0

    def __next__(self):
        if self._index >= len(self._keys):
            raise StopIteration
        key = self._keys[self._index]
        if self._mode == "keys":
            value = key
        elif self._mode == "values":
            value = self.var[key]
        elif self._mode == "items":
            value = (key, self.var[key])
        else:
            raise ValueError("Invalid iterator type")
        self._index += 1
        return value

    def __iter__(self):
        return self

    def __repr__(self):
        return f"TrackedDictIteratorProxy({self.var}, mode='{self._mode}'))"


# Iterator Class for TrackedDictProxy.keys
class TrackedDictKeysIteratorProxy(TrackedDictIteratorProxy):
    def __init__(self, v):
        super().__init__(v, mode="keys")

    def __repr__(self):
        return f"TrackedDictKeysIteratorProxy({self._keys})"


# Iterator Class for TrackedDictProxy.values
class TrackedDictValuesIteratorProxy(TrackedDictIteratorProxy):
    def __init__(self, v):
        super().__init__(v, mode="values")

    def __repr__(self):
        return f"TrackedDictValuesIteratorProxy({self._values})"


# Iterator Class for TrackedDictProxy.items
class TrackedDictItemsIteratorProxy(TrackedDictIteratorProxy):
    def __init__(self, v):
        super().__init__(v, mode="items")

    def __repr__(self):
        return f"TrackedDictItemsIteratorProxy({self._items})"


# Iterator Class for TrackedBytesProxy
class TrackedBytesIteratorProxy(TrackedVarProxy):
    def __init__(self, v):
        self.var = v
        self._index = 0

    def __next__(self):
        if self._index >= len(self.var):
            raise StopIteration
        value = self.var[self._index]
        self._index += 1
        return value

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.var)

    def __reversed__(self):
        return TrackedBytesIteratorProxy(reversed(self.var))

    def __contains__(self, item):
        return item in self.var

    def __repr__(self):
        return f"TrackedBytesIteratorProxy({self.var})"


"""Configs"""

NON_WRAPPED_METHODS = [
    "get_var",
    "override_dtype_check",
    "__init__",
    "__new__",
    "__hash__",
    "__class__",
    "__copy__",
    "__deepcopy__",
    "__subclasshook__",
    "__init_subclass__",
]

AVAILABLE_RAW_RET = [
    "__float__",
    "__len__",
    "__int__",
]

RAW_RET_METHODS = [
    "__bool__",
    "__float__",
    "__format__",
    "__len__",
    "__index__",
    "__int__",
    "__repr__",
    "__str__",
    "__contains__",
]

INPLACE_METHODS_WITHOUT_RET = [
    "__delitem__",
    "__iadd__",
    "__imul__",
    "__setitem__",
    "extend",
    "append",
    "clear",
    "insert",
    # "pop",
    "remove",
    "reverse",
    "sort",
    # "setdefault",
    "update",
]

ATTRIBUTES = [
    "denominator",
    "numerator",
    "imag",
    "real",
]

# Use the `type_to_proxy()` as an interface to
# retreive this dictionary. Do not use it directly.
BUILTIN_TYPE_TO_PROXY = {
    "int": TrackedIntProxy,
    "float": TrackedFloatProxy,
    "str": TrackedStrProxy,
    "list": TrackedListProxy,
    "tuple": TrackedTupleProxy,
    "dict": TrackedDictProxy,
    "bytes": TrackedBytesProxy,
}

# Use the `type_to_proxy()` as an interface to
# retreive this dictionary. Do not use it directly.
TYPE_TO_PROXY = {
    **BUILTIN_TYPE_TO_PROXY,
    "Enum": TrackedEnumProxy,
    "IntEnum": TrackedIntEnumProxy,
    "Size": TrackedTupleProxy,
    "Shape": TrackedShapeProxy,
    "TensorShape": TrackedTensorShapeProxy,
}

PROXY_ITERATOR_TO_BUILTIN_TYPES = {
    TrackedStrIteratorProxy: "str",
    TrackedTupleIteratorProxy: "tuple",
    TrackedListIteratorProxy: "list",
    TrackedDictIteratorProxy: "dict",
    TrackedDictKeysIteratorProxy: "dict",
    TrackedDictValuesIteratorProxy: "dict",
    TrackedDictItemsIteratorProxy: "dict",
    TrackedBytesIteratorProxy: "bytes",
}

PROXY_ITERATOR_TO_TYPES = {
    **PROXY_ITERATOR_TO_BUILTIN_TYPES,
    TrackedShapeIteratorProxy: "Shape",
}

PROXY_TYPE_TO_ITERATOR_PROXY_TYPE = {
    TrackedStrProxy: TrackedStrIteratorProxy,
    TrackedTupleProxy: TrackedTupleIteratorProxy,
    TrackedListProxy: TrackedListIteratorProxy,
    TrackedDictProxy: TrackedDictIteratorProxy,
    TrackedBytesProxy: TrackedBytesIteratorProxy,
    TrackedShapeProxy: TrackedShapeIteratorProxy,
}

BUILTIN_ITERATOR_METHODS = ["__iter__", "__next__"]

DICT_ITERATOR_METHODS = ["keys", "values", "items"]

ITERATOR_METHODS = [*DICT_ITERATOR_METHODS, *BUILTIN_ITERATOR_METHODS]

# Note: This explicitly wraps and copies over attributes from `mapping` to `key` if
# the attribute is in `to_map` and not in `to_ignore`.
# 1. "*" means all. If "*" in `to_map`, map all attrs except ones in `to_ignore`.
# 2. If "*" in `to_ignore`, ignore all attrs except ones in `to_map`. Vice versa.
# 3. If `to_map` is empty, map no attrs. If `to_ignore` is empty, ignore no attrs.
ATTRS_TO_WRAP_AND_MAP = {
    "TrackedIntEnumProxy": {
        "mapping": int,
        "to_map": ["*"],
        "to_ignore": [
            "__getattribute__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
            "__dir__",
            "__index__",
            "__init__",
            "__init_subclass__",
            "__hash__",
            "__subclasshook__",
            "__class__",
            "__new__",
            "__getnewargs__",
        ],
    },
}

TRACKED_VAR_PROXY_META_CLASSES = [
    "TrackedTensorShapeProxyMeta",
]

PROXY_TO_BUILTIN_TYPES = {v: k for k, v in BUILTIN_TYPE_TO_PROXY.items()}

# set the class names, so Keras 3 will correctly infer dtype
# needed for tests/tracked_var_proxy/test_tracked_var_proxy.py::test_batch_keras with tf>=2.16.0
TrackedFloatProxy.__name__ = "float64"
TrackedIntProxy.__name__ = "int64"


"""Convenient attrs"""
tracked_var_proxy_iter_classes = set([cls.__name__ for cls in PROXY_ITERATOR_TO_TYPES])
_tf_tensor_shape_override = False
