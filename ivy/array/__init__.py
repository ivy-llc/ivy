# global
import copy
import functools

# local
from . import array_mode_handler
from .array_mode_handler import *
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


class Array(ArrayWithDevice, ArrayWithGeneral, ArrayWithGradients, ArrayWithImage, ArrayWithLinalg, ArrayWithLogic,
            ArrayWithMath, ArrayWithMeta, ArrayWithRandom, ArrayWithReductions):

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
        return self._data.__dir__()

    @_native_wrapper
    def __getattr__(self, item):
        try:
            attr = self._data.__getattr__(item)
        except AttributeError:
            attr = self._data.__getattribute__(item)
        return to_ivy(attr)

    @_native_wrapper
    def __getitem__(self, query):
        return to_ivy(self._data.__getitem__(query))

    @_native_wrapper
    def __setitem__(self, query, val):
        self._data.__setitem__(query, val)

    @_native_wrapper
    def __contains__(self, key):
        return self._data.__contains__(key)

    @_native_wrapper
    def __pos__(self):
        return self

    @_native_wrapper
    def __neg__(self):
        res = self._data.__neg__()
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __pow__(self, power):
        power = to_native(power)
        res = self._data.__pow__(power)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rpow__(self, power):
        power = to_native(power)
        res = self._data.__rpow__(power)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __add__(self, other):
        other = to_native(other)
        res = self._data.__add__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __radd__(self, other):
        other = to_native(other)
        res = self._data.__radd__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __sub__(self, other):
        other = to_native(other)
        res = self._data.__sub__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rsub__(self, other):
        other = to_native(other)
        res = self._data.__rsub__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __mul__(self, other):
        other = to_native(other)
        res = self._data.__mul__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rmul__(self, other):
        other = to_native(other)
        res = self._data.__rmul__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __truediv__(self, other):
        other = to_native(other)
        res = self._data.__truediv__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rtruediv__(self, other):
        other = to_native(other)
        res = self._data.__rtruediv__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __floordiv__(self, other):
        other = to_native(other)
        res = self._data.__floordiv__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rfloordiv__(self, other):
        other = to_native(other)
        res = self._data.__rfloordiv__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __abs__(self):
        res = self._data.__abs__()
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __float__(self):
        res = self._data.__float__()
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __int__(self):
        res = self._data.__int__()
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __bool__(self):
        return self._data.__bool__()

    @_native_wrapper
    def __lt__(self, other):
        other = to_native(other)
        res = self._data.__lt__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __le__(self, other):
        other = to_native(other)
        res = self._data.__le__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __eq__(self, other):
        other = to_native(other)
        res = self._data.__eq__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __ne__(self, other):
        other = to_native(other)
        res = self._data.__ne__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __gt__(self, other):
        other = to_native(other)
        res = self._data.__gt__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __ge__(self, other):
        other = to_native(other)
        res = self._data.__ge__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __and__(self, other):
        other = to_native(other)
        res = self._data.__and__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rand__(self, other):
        other = to_native(other)
        res = self._data.__rand__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __or__(self, other):
        other = to_native(other)
        res = self._data.__or__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __ror__(self, other):
        other = to_native(other)
        res = self._data.__ror__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __invert__(self):
        res = self._data.__invert__()
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __xor__(self, other):
        other = to_native(other)
        res = self._data.__xor__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rxor__(self, other):
        other = to_native(other)
        res = self._data.__rxor__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    # noinspection PyDefaultArgument
    @_native_wrapper
    def __deepcopy__(self, memodict={}):
        try:
            return to_ivy(self._data.__deepcopy__(memodict))
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
