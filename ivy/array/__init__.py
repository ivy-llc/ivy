# global
import copy
import functools
from numbers import Number
from operator import mul
# local
import ivy
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
        if ivy.is_ivy_array(data):
            self._data = data.data
        else:
            assert ivy.is_native_array(data)
            self._data = data
        self._shape = self._data.shape
        self._size = functools.reduce(mul, self._data.shape) if len(self._data.shape) > 0 else 0
        self._dtype = ivy.dtype(self._data)
        self._device = ivy.dev(self._data)
        self._dev_str = ivy.dev_to_str(self._device)
        self._pre_repr = 'ivy.'
        if 'gpu' in self._dev_str:
            self._post_repr = ', dev={})'.format(self._dev_str)
        else:
            self._post_repr = ')'

    # Properties #
    # -----------#

    @property
    def mT(self):
        assert len(self._data.shape) >= 2
        return ivy.matrix_transpose(self._data)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        """
        Number of array dimensions (axes).
        Returns
        -------
        out: int
            number of array dimensions (axes).
        """
        return len(tuple(self._shape))

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    # noinspection PyPep8Naming
    @property
    def T(self):
        assert len(self._data.shape) == 2
        return ivy.matrix_transpose(self._data)

    @property
    def size(self):
        """
        Number of elements in an array.
        
        .. note::
           This must equal the product of the array's dimensions.
        
        Returns
        -------
        out: Optional[int]
            number of elements in an array. The returned value must be ``None`` if and only if one or more array dimensions are unknown.
        
        
        .. note::
           For array libraries having graph-based computational models, an array may have unknown dimensions due to data-dependent operations.
        """
        return self._size
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
            attr = self._data.__getattribute__(item)
        except AttributeError:
            attr = self._data.__getattr__(item)
        return to_ivy(attr)

    @_native_wrapper
    def __getitem__(self, query):
        query = to_native(query)
        return to_ivy(self._data.__getitem__(query))

    @_native_wrapper
    def __setitem__(self, query, val):
        try:
            self._data.__setitem__(query, val)
        except (AttributeError, TypeError):
            self._data = ivy.scatter_nd(query, val, tensor=self._data, reduction='replace')

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
        return ivy.pow(self._data, power)

    @_native_wrapper
    def __rpow__(self, power):
        return self._data.__rpow__(power)

    @_native_wrapper
    def __add__(self, other):
        other = to_native(other)
        return ivy.add(self._data, other)

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
        res = ivy.subtract(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __rsub__(self, other):
        other = to_native(other)
        res = -ivy.subtract(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __mul__(self, other):
        other = to_native(other)
        res = ivy.multiply(self._data, other)
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
    def __mod__(self, other):
        return ivy.remainder(self._data, other)

    @_native_wrapper
    def __truediv__(self, other):
        return ivy.divide(self._data, other)

    @_native_wrapper
    def __rtruediv__(self, other):
        other = to_native(other)
        res = self._data.__rtruediv__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __floordiv__(self, other):
        return ivy.floor_divide(self._data, other)

    @_native_wrapper
    def __rfloordiv__(self, other):
        other = to_native(other)
        res = self._data.__rfloordiv__(other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __abs__(self):
        if 'uint' in ivy.dtype(self._data, as_str=True):
            return self
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
        if hasattr(self._data, '__int__'):
            res = self._data.__int__()
        else:
            # noinspection PyTypeChecker
            res = int(ivy.to_scalar(self._data))
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __bool__(self):
        return self._data.__bool__()

    @_native_wrapper
    def __lt__(self, other):
        other = to_native(other)
        res = ivy.less(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __le__(self, other):
        return ivy.less_equal(self._data, other)

    @_native_wrapper
    def __eq__(self, other):
        other = to_native(other)
        res = ivy.equal(self._data, other)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    @_native_wrapper
    def __ne__(self, other):
        return ivy.not_equal(self._data, other)

    @_native_wrapper
    def __gt__(self, other):
        return ivy.greater(self._data, other)

    @_native_wrapper
    def __ge__(self, other):
        return ivy.greater_equal(self._data, other)

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
        return ivy.bitwise_or(self._data, other)


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
        res = ivy.bitwise_xor(self._data, other)
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

    @_native_wrapper
    def __rshift__(self, other):
        return ivy.bitwise_right_shift(self._data, other)

    @_native_wrapper
    def __rrshift__(self, other):
        other = to_native(other)
        res = self._data.__rrshift__(other)
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
