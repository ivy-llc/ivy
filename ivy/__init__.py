import ivy
from .core import *
from . import neural_net_functional
from .neural_net_functional import *
from . import neural_net_stateful
from .neural_net_stateful import *
from . import verbosity
from .framework_handler import current_framework, get_framework, set_framework, unset_framework, framework_stack,\
    set_debug_mode, set_breakpoint_debug_mode, set_exception_debug_mode, unset_debug_mode, debug_mode, debug_mode_val,\
    set_wrapped_mode, unset_wrapped_mode, wrapped_mode, wrapped_mode_val

_MIN_DENOMINATOR = 1e-12
_MIN_BASE = 1e-5
NativeArray = None


class Array:

    def __init__(self, data):
        assert ivy.is_array(data)
        self._data = data
        self._pre_repr = 'ivy.'
        self._post_repr = ', backend={})'

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return ivy.dtype(self._data)

    # Built-ins #
    # ----------#

    def __array__(self, *args, **kwargs):
        raw_args = [a.data if isinstance(a, ivy.Array) else a for a in args]
        raw_kwargs = dict([(k, v.data if isinstance(v, ivy.Array) else v) for k, v in kwargs.items()])
        return self._data.__array__(*raw_args, **raw_kwargs)

    def __array_prepare__(self, *args, **kwargs):
        raw_args = [a.data if isinstance(a, ivy.Array) else a for a in args]
        raw_kwargs = dict([(k, v.data if isinstance(v, ivy.Array) else v) for k, v in kwargs.items()])
        return self._data.__array_prepare__(*raw_args, **raw_kwargs)

    def __array_ufunc__(self, *args, **kwargs):
        raw_args = [a.data if isinstance(a, ivy.Array) else a for a in args]
        raw_kwargs = dict([(k, v.data if isinstance(v, ivy.Array) else v) for k, v in kwargs.items()])
        return self._data.__array_ufunc__(*raw_args, **raw_kwargs)

    def __array_wrap__(self, *args, **kwargs):
        raw_args = [a.data if isinstance(a, ivy.Array) else a for a in args]
        raw_kwargs = dict([(k, v.data if isinstance(v, ivy.Array) else v) for k, v in kwargs.items()])
        return self._data.__array_wrap__(*raw_args, **raw_kwargs)

    def __repr__(self):
        return self._pre_repr + ivy.to_numpy(self._data).__repr__()[:-1].replace('\n', '\n    ') + \
               self._post_repr.format(ivy.current_framework_str())

    def __dir__(self):
        return self._data.__dir__()

    def __getattr__(self, item):
        return self._data.__getattr__(item)

    def __getitem__(self, query):
        return self._data.__getitem__(query)

    def __setitem__(self, query, val):
        return self._data.__setitem__(query, val)

    def __contains__(self, key):
        return self._data.__contains__(key)

    def __pos__(self):
        return self

    def __neg__(self):
        res = self._data.__neg__()
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __pow__(self, power):
        power = ivy.as_native(power)
        res = self._data.__pow__(power)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __rpow__(self, power):
        power = ivy.as_native(power)
        res = self._data.__rpow__(power)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __add__(self, other):
        other = ivy.as_native(other)
        res = self._data.__add__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __radd__(self, other):
        other = ivy.as_native(other)
        res = self._data.__radd__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __sub__(self, other):
        other = ivy.as_native(other)
        res = self._data.__sub__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __rsub__(self, other):
        other = ivy.as_native(other)
        res = self._data.__rsub__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __mul__(self, other):
        other = ivy.as_native(other)
        res = self._data.__mul__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __rmul__(self, other):
        other = ivy.as_native(other)
        res = self._data.__rmul__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __truediv__(self, other):
        other = ivy.as_native(other)
        res = self._data.__truediv__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __rtruediv__(self, other):
        other = ivy.as_native(other)
        res = self._data.__rtruediv__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __floordiv__(self, other):
        other = ivy.as_native(other)
        res = self._data.__floordiv__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __rfloordiv__(self, other):
        other = ivy.as_native(other)
        res = self._data.__rfloordiv__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __abs__(self):
        res = self._data.__abs__()
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __bool__(self):
        return self._data.__bool__()

    def __lt__(self, other):
        other = ivy.as_native(other)
        res = self._data.__lt__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __le__(self, other):
        other = ivy.as_native(other)
        res = self._data.__le__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __eq__(self, other):
        other = ivy.as_native(other)
        res = self._data.__eq__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __ne__(self, other):
        other = ivy.as_native(other)
        res = self._data.__ne__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __gt__(self, other):
        other = ivy.as_native(other)
        res = self._data.__gt__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __ge__(self, other):
        other = ivy.as_native(other)
        res = self._data.__ge__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __and__(self, other):
        other = ivy.as_native(other)
        res = self._data.__and__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __rand__(self, other):
        other = ivy.as_native(other)
        res = self._data.__rand__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __or__(self, other):
        other = ivy.as_native(other)
        res = self._data.__or__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __ror__(self, other):
        other = ivy.as_native(other)
        res = self._data.__ror__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __invert__(self):
        res = self._data.__invert__()
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __xor__(self, other):
        other = ivy.as_native(other)
        res = self._data.__xor__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res

    def __rxor__(self, other):
        other = ivy.as_native(other)
        res = self._data.__rxor__(other)
        if res is NotImplemented:
            return res
        return Array(res) if ivy.is_array(res) else res


class Variable(Array):

    def __init__(self, data):
        assert ivy.is_variable(data)
        Array.__init__(self, data)


class Framework:

    def __init__(self):
        pass


class Device:

    def __init__(self):
        pass


class Dtype:

    def __init__(self):
        pass


backend = 'none'
