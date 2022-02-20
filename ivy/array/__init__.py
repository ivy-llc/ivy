# global
import copy
import functools
from numbers import Number

# local
from . import array_api
from .array_api import *
from . import conversions
from .conversions import *
from . import device
from .device import *
from . import general
from .general import *
from . import gradients
from .gradients import *
from . import image
from .image import *
from . import linalg
from .linalg import *
from . import logic
from .logic import *
from . import math
from .math import *
from . import meta
from .meta import *
from . import random
from .random import *
from . import reductions
from .reductions import *


def _native_wrapper(f):
    @functools.wraps(f)
    def decor(self, *args, **kwargs):
        if isinstance(self, Array):
            return f(self, *args, **kwargs)
        return getattr(self, f.__name__)(*args, **kwargs)
    return decor


class Array(ArrayWithArrayAPI, ArrayWithDevice, ArrayWithGeneral, ArrayWithGradients, ArrayWithImage, ArrayWithLinalg,
            ArrayWithLogic, ArrayWithMath, ArrayWithMeta, ArrayWithRandom, ArrayWithReductions):

    def __init__(self, data):
        assert ivy.is_array(data)
        self._data = data
        self._shape = data.shape
        self._dtype = ivy.dtype(self._data)
        self._device = ivy.dev(data)
        self._dev_str = ivy.dev_to_str(self._device)
        self._pre_repr = 'ivy.'
        if 'gpu' in self._dev_str:
            self._post_repr = ', dev={})'.format(self._dev_str)
        else:
            self._post_repr = ')'

    # Properties #
    # -----------#

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    # Built-ins #
    # ----------#

    @_native_wrapper
    def __array__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array__(*args, **kwargs)

    @_native_wrapper
    def __array_prepare__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_prepare__(*args, **kwargs)

    @_native_wrapper
    def __array_ufunc__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_ufunc__(*args, **kwargs)

    @_native_wrapper
    def __array_wrap__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_wrap__(*args, **kwargs)

    @_native_wrapper
    def __repr__(self):
        return self._pre_repr + ivy.to_numpy(self._data).__repr__()[:-1].replace('\n', '\n    ') + \
               self._post_repr.format(ivy.current_framework_str())

    @_native_wrapper
    def __dir__(self):
        return ivy.builtin_dir(self._data)

    @_native_wrapper
    def __getattr__(self, item):
        try:
            attr = ivy.builtin_getattribute(self._data, item)
        except AttributeError:
            attr = ivy.builtin_getattr(self._data, item)
        return to_ivy(attr)

    @_native_wrapper
    def __getitem__(self, query):
        return to_ivy(ivy.builtin_getitem(self._data, query))

    @_native_wrapper
    def __setitem__(self, query, val):
        try:
            ivy.builtin_setitem(self._data, query, val)
        except (AttributeError, TypeError):
            query = [[query]] if isinstance(query, Number) else query
            query = ivy.array(query)
            if len(query.shape) < 2:
                query = ivy.expand_dims(query, -1)
            val = [val] if isinstance(val, Number) else val
            val = ivy.array(val, dtype=ivy.dtype(self._data))
            self._data = ivy.scatter_nd(query, val, tensor=self._data, reduction='replace')

    @_native_wrapper
    def __contains__(self, key):
        return ivy.builtin_contains(self._data, key)

    @_native_wrapper
    def __pos__(self):
        return self

    @_native_wrapper
    def __neg__(self):
        res = ivy.builtin_neg(self._data)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __pow__(self, power):
        power = to_native(power)
        res = ivy.builtin_pow(self._data, power)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rpow__(self, power):
        power = to_native(power)
        res = ivy.builtin_rpow(self._data, power)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __add__(self, other):
        other = to_native(other)
        res = ivy.builtin_add(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __radd__(self, other):
        other = to_native(other)
        res = ivy.builtin_radd(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __sub__(self, other):
        other = to_native(other)
        res = ivy.builtin_sub(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rsub__(self, other):
        other = to_native(other)
        res = ivy.builtin_rsub(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __mul__(self, other):
        other = to_native(other)
        res = ivy.builtin_mul(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rmul__(self, other):
        other = to_native(other)
        res = ivy.builtin_rmul(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __truediv__(self, other):
        other = to_native(other)
        res = ivy.builtin_truediv(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rtruediv__(self, other):
        other = to_native(other)
        res = ivy.builtin_rtruediv(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __floordiv__(self, other):
        other = to_native(other)
        res = ivy.builtin_floordiv(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rfloordiv__(self, other):
        other = to_native(other)
        res = ivy.builtin_rfloordiv(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __abs__(self):
        if 'uint' in ivy.dtype(self._data, as_str=True):
            return self
        res = ivy.builtin_abs(self._data)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __float__(self):
        res = ivy.builtin_float(self._data)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __int__(self):
        if hasattr(self._data, '__int__'):
            res = ivy.builtin_int(self._data)
        else:
            # noinspection PyTypeChecker
            res = int(ivy.to_scalar(self._data))
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __bool__(self):
        return ivy.builtin_bool(self._data)

    @_native_wrapper
    def __lt__(self, other):
        other = to_native(other)
        res = ivy.builtin_lt(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __le__(self, other):
        other = to_native(other)
        res = ivy.builtin_le(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __eq__(self, other):
        other = to_native(other)
        res = ivy.builtin_eq(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __ne__(self, other):
        other = to_native(other)
        res = ivy.builtin_ne(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __gt__(self, other):
        other = to_native(other)
        res = ivy.builtin_gt(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __ge__(self, other):
        other = to_native(other)
        res = ivy.builtin_ge(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __and__(self, other):
        other = to_native(other)
        res = ivy.builtin_and(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rand__(self, other):
        other = to_native(other)
        res = ivy.builtin_rand(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __or__(self, other):
        other = to_native(other)
        res = ivy.builtin_or(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __ror__(self, other):
        other = to_native(other)
        res = ivy.builtin_ror(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __invert__(self):
        res = ivy.builtin_invert(self._data)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __xor__(self, other):
        other = to_native(other)
        res = ivy.builtin_xor(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rxor__(self, other):
        other = to_native(other)
        res = ivy.builtin_rxor(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    # noinspection PyDefaultArgument
    @_native_wrapper
    def __deepcopy__(self, memodict={}):
        try:
            return to_ivy(ivy.builtin_deepcopy(self._data, memodict))
        except AttributeError:
            # ToDo: try and find more elegant solution to jax inability to deepcopy device arrays
            if ivy.current_framework_str() == 'jax':
                np_array = copy.deepcopy(self._data)
                jax_array = ivy.array(np_array)
                return to_ivy(jax_array)
            return to_ivy(copy.deepcopy(self._data))

    @_native_wrapper
    def __iter__(self):
        return iter([to_ivy(i) for i in self._data])


# noinspection PyRedeclaration
class Variable(Array):

    def __init__(self, data):
        assert ivy.is_variable(data)
        super().__init__(data)
