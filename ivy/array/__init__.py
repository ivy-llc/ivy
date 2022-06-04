# global
import copy
import pickle
import functools
from operator import mul

# local
from . import conversions
from .conversions import *

from .activations import ArrayWithActivations
from .creation import ArrayWithCreation
from .data_types import ArrayWithDataTypes
from .device import ArrayWithDevice
from .elementwise import ArrayWithElementwise
from .general import ArrayWithGeneral
from .gradients import ArrayWithGradients
from .image import ArrayWithImage
from .layers import ArrayWithLayers
from .linear_algebra import ArrayWithLinearAlgebra
from .losses import ArrayWithLosses
from .manipulation import ArrayWithManipulation
from .norms import ArrayWithNorms
from .random import ArrayWithRandom
from .searching import ArrayWithSearching
from .set import ArrayWithSet
from .sorting import ArrayWithSorting
from .statistical import ArrayWithStatistical
from .utility import ArrayWithUtility
from .wrapping import add_ivy_array_instance_methods


def _native_wrapper(f):
    @functools.wraps(f)
    def decor(self, *args, **kwargs):
        if isinstance(self, Array):
            return f(self, *args, **kwargs)
        return getattr(self, f.__name__)(*args, **kwargs)

    return decor


class Array(
    ArrayWithActivations,
    ArrayWithCreation,
    ArrayWithDataTypes,
    ArrayWithDevice,
    ArrayWithElementwise,
    ArrayWithGeneral,
    ArrayWithGradients,
    ArrayWithImage,
    ArrayWithLayers,
    ArrayWithLinearAlgebra,
    ArrayWithLosses,
    ArrayWithManipulation,
    ArrayWithNorms,
    ArrayWithRandom,
    ArrayWithSearching,
    ArrayWithSet,
    ArrayWithSorting,
    ArrayWithStatistical,
    ArrayWithUtility,
):
    def __init__(self, data):
        ArrayWithActivations.__init__(self)
        ArrayWithCreation.__init__(self)
        ArrayWithDataTypes.__init__(self)
        ArrayWithDevice.__init__(self)
        ArrayWithElementwise.__init__(self)
        ArrayWithGeneral.__init__(self)
        ArrayWithGradients.__init__(self)
        ArrayWithImage.__init__(self)
        ArrayWithLayers.__init__(self)
        ArrayWithLinearAlgebra.__init__(self)
        ArrayWithLosses.__init__(self)
        ArrayWithManipulation.__init__(self)
        ArrayWithNorms.__init__(self)
        ArrayWithRandom.__init__(self)
        ArrayWithSearching.__init__(self)
        ArrayWithSet.__init__(self)
        ArrayWithSorting.__init__(self)
        ArrayWithStatistical.__init__(self)
        ArrayWithUtility.__init__(self)
        if ivy.is_ivy_array(data):
            self._data = data.data
        else:
            assert ivy.is_native_array(data)
            self._data = data
        self._shape = self._data.shape
        self._size = (
            functools.reduce(mul, self._data.shape) if len(self._data.shape) > 0 else 0
        )
        self._dtype = ivy.dtype(self._data)
        self._device = ivy.dev(self._data)
        self._dev_str = ivy.as_ivy_dev(self._device)
        self._pre_repr = "ivy."
        if "gpu" in self._dev_str:
            self._post_repr = ", dev={})".format(self._dev_str)
        else:
            self._post_repr = ")"
        
        self.framework_str = ivy.current_framework_str()

    # Properties #
    # -----------#

    # noinspection PyPep8Naming
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
        """Number of elements in an array.

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

    # Setters #
    # --------#

    @data.setter
    def data(self, data):
        assert ivy.is_native_array(data)
        self._data = data

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
        return (
            self._pre_repr
            + ivy.to_numpy(self._data).__repr__()[:-1].replace("\n", "\n    ")
            + self._post_repr.format(ivy.current_framework_str())
        )

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
            self._data = ivy.scatter_nd(
                query, val, tensor=self._data, reduction="replace"
            )._data
            self._dtype = ivy.dtype(self._data)

    @_native_wrapper
    def __contains__(self, key):
        return self._data.__contains__(key)

    @_native_wrapper
    def __getstate__(self):
        data_dict = dict()

        # only pickle the native array
        data_dict['data'] = self.data

        # also store the local ivy framework that created this array
        data_dict['framework_str'] = self.framework_str
        data_dict['device_str'] = ivy.as_ivy_dev(self.device)

        return data_dict

    @_native_wrapper
    def __setstate__(self, state):
        # we can construct other details of ivy.Array 
        # just by re-creating the ivy.Array using the native array

        # get the required backend
        backend = ivy.get_framework(state['framework_str'])
        ivy_array = backend.array(state['data'])

        # TODO: what about placement of the array on the right device ?
        device = backend.dev_from_str(state['device_str'])

        self.__dict__ = ivy_array.__dict__

        backend.to_dev(self, device)

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
        return ivy.subtract(self._data, other)

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
        if "uint" in ivy.dtype(self._data, as_str=True):
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
        if hasattr(self._data, "__int__"):
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
        return ivy.bitwise_and(self._data, other)

    @_native_wrapper
    def __rand__(self, other):
        return ivy.bitwise_and(self._data, other)

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
    def __lshift__(self, other):
        return ivy.bitwise_left_shift(self._data, other)

    @_native_wrapper
    def __rlshift__(self, other):
        other = to_native(other)
        res = self._data.__rlshift__(other)
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
            if ivy.current_framework_str() == "jax":
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
