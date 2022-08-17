# flake8: noqa
# global
import copy
import functools
import numpy as np
from operator import mul

# local
import ivy
from .conversions import *
from .activations import ArrayWithActivations
from .creation import ArrayWithCreation
from .data_type import ArrayWithDataTypes
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
        self._init(data)

    def _init(self, data):
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
        self.framework_str = ivy.current_backend_str()
        self._is_variable = ivy.is_variable(self._data)

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
        return ivy.Shape(self._shape)

    @property
    def ndim(self):
        """Number of array dimensions (axes).

        Returns
        -------
        ret
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
        out
            number of elements in an array. The returned value must be ``None`` if
            and only if one or more array dimensions are unknown.


        .. note::
           For array libraries having graph-based computational models, an array may
           have unknown dimensions due to data-dependent operations.

        """
        return self._size

    @property
    def is_variable(self):
        """Determines whether the array is a variable or not.

        Returns
        -------
        ret
            Boolean, true if the array is a trainable variable, false otherwise.
        """
        return self._is_variable

    # Setters #
    # --------#

    @data.setter
    def data(self, data):
        assert ivy.is_native_array(data)
        self._init(data)

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
        sig_fig = ivy.array_significant_figures()
        dec_vals = ivy.array_decimal_values()
        rep = (
            ivy.vec_sig_fig(ivy.to_numpy(self._data), sig_fig)
            if self._size > 0
            else ivy.to_numpy(self._data)
        )
        with np.printoptions(precision=dec_vals):
            return (
                self._pre_repr
                + rep.__repr__()[:-1].partition(", dtype")[0].partition(", dev")[0]
                + self._post_repr.format(ivy.current_backend_str())
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
        data_dict["data"] = self.data

        # also store the local ivy framework that created this array
        data_dict["framework_str"] = self.framework_str
        data_dict["device_str"] = ivy.as_ivy_dev(self.device)

        return data_dict

    @_native_wrapper
    def __setstate__(self, state):
        # we can construct other details of ivy.Array
        # just by re-creating the ivy.Array using the native array

        # get the required backend
        ivy.set_backend(state["framework_str"])
        ivy_array = ivy.array(state["data"])
        ivy.unset_backend()

        self.__dict__ = ivy_array.__dict__

        # TODO: what about placement of the array on the right device ?
        # device = backend.as_native_dev(state["device_str"])
        # backend.to_device(self, device)

    @_native_wrapper
    def __pos__(self):
        return ivy.positive(self._data)

    @_native_wrapper
    def __neg__(self):
        return ivy.negative(self._data)

    @_native_wrapper
    def __pow__(self, power):
        return ivy.pow(self._data, power)

    @_native_wrapper
    def __rpow__(self, power):
        return self._data.__rpow__(power)

    @_native_wrapper
    def __ipow__(self, power):
        return ivy.pow(self._data, power)

    @_native_wrapper
    def __add__(self, other):
        """
        ivy.Array special method variant of ivy.add. This method simply wraps the
        function, and so the docstring for ivy.add also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        other
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Array` instances only:

        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([4, 5, 6])
        >>> z = x + y
        >>> print(z)
        ivy.array([5, 7, 9])

        With mix of :code:`ivy.Array` and :code:`ivy.Container` instances:

        >>> x = ivy.array([[1.1, 2.3, -3.6]])
        >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),\
                            b=ivy.array([[5.], [6.], [7.]]))
        >>> z = x + y
        >>> print(z)
        {
            a: ivy.array([[5.1, 6.3, 0.4],
                          [6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4]]),
            b: ivy.array([[6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4],
                          [8.1, 9.3, 3.4]])
        }
        """
        return ivy.add(self._data, other)

    @_native_wrapper
    def __radd__(self, other):
        """
        ivy.Array reverse special method variant of ivy.add. This method simply wraps
        the function, and so the docstring for ivy.add also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        other
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = 1
        >>> y = ivy.array([4, 5, 6])
        >>> z = x + y
        >>> print(z)
        ivy.array([5, 6, 7])
        """
        return ivy.add(other, self._data)

    @_native_wrapper
    def __iadd__(self, other):
        return ivy.add(self._data, other)

    @_native_wrapper
    def __sub__(self, other):
        return ivy.subtract(self._data, other)

    @_native_wrapper
    def __rsub__(self, other):
        return ivy.subtract(other, self._data)

    @_native_wrapper
    def __mul__(self, other):
        return ivy.multiply(self._data, other)

    @_native_wrapper
    def __rmul__(self, other):
        return ivy.multiply(other, self._data)

    @_native_wrapper
    def __imul__(self, other):
        return ivy.multiply(self._data, other)

    @_native_wrapper
    def __mod__(self, other):
        return ivy.remainder(self._data, other)

    @_native_wrapper
    def __imod__(self, other):
        return ivy.remainder(self._data, other)

    @_native_wrapper
    def __truediv__(self, other):
        return ivy.divide(self._data, other)

    @_native_wrapper
    def __rtruediv__(self, other):
        return ivy.divide(other, self._data)

    @_native_wrapper
    def __itruediv__(self, other):
        return ivy.divide(self._data, other)

    @_native_wrapper
    def __floordiv__(self, other):
        return ivy.floor_divide(self._data, other)

    @_native_wrapper
    def __rfloordiv__(self, other):
        return ivy.floor_divide(other, self._data)

    @_native_wrapper
    def __ifloordiv__(self, other):
        return ivy.floor_divide(self._data, other)

    @_native_wrapper
    def __abs__(self):
        return ivy.abs(self._data)

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
        return ivy.less(self._data, other)

    @_native_wrapper
    def __le__(self, other):
        """
        Less than or equal to

        Returns
        -------
        an array containing the element-wise results. The returned array must have a
        data type of bool.

        Operator Examples
        -----------------

        With :code:`ivy.Array` instances:

        >>> x = ivy.array([6, 2, 3])
        >>> y = ivy.array([4, 5, 6])
        >>> z = x <= y
        >>> print(z)
        ivy.array([ False, True, True])

        With :code:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),\
                      b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 2, 3]),\
                          b=ivy.array([5, 6, 7]))
        >>> z = x <= y
        >>> print(z)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([True, True, True])
        }

        With mix of :code:`ivy.Array` and :code:`ivy.Container` instances:

        >>> x = ivy.array([[5.1, 2.3, -3.6]])
        >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),\
                              b=ivy.array([[5.], [6.], [7.]]))
        >>> z = x <= y
        >>> print(z)
        {
            a: ivy.array([[False, True, True],
                          [False, True, True],
                          [True, True, True]]),
            b: ivy.array([[False, True, True],
                          [True, True, True],
                          [True, True, True]])
        }
        """
        return ivy.less_equal(self._data, other)

    @_native_wrapper
    def __eq__(self, other):
        return ivy.equal(self._data, other)

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
        return ivy.bitwise_and(other, self._data)

    @_native_wrapper
    def __or__(self, other):
        return ivy.bitwise_or(self._data, other)

    @_native_wrapper
    def __ror__(self, other):
        return ivy.bitwise_or(other, self._data)

    @_native_wrapper
    def __invert__(self):
        return ivy.bitwise_invert(self._data)

    @_native_wrapper
    def __xor__(self, other):
        return ivy.bitwise_xor(self._data, other)

    @_native_wrapper
    def __rxor__(self, other):
        return ivy.bitwise_xor(other, self._data)

    @_native_wrapper
    def __lshift__(self, other):
        return ivy.bitwise_left_shift(self._data, other)

    @_native_wrapper
    def __rlshift__(self, other):
        return ivy.bitwise_left_shift(other, self._data)

    @_native_wrapper
    def __rshift__(self, other):
        """
        ivy.Array special method variant of ivy.bitwise_right_shift. This method 
        simply wraps the function, and so the docstring for ivy.bitwise_right_shift
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer data type.
        other
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). 
            Should have an integer data type. Each element must be greater than or equal
            to ``0``.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have
            a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Array` instances only:

        >>> a = ivy.array([2, 3, 4])
        >>> b = ivy.array([0, 1, 2])
        >>> y = a >> b
        >>> print(y)
        ivy.array([2, 1, 1])

        With mix of :code:`ivy.Array` and :code:`ivy.Container` instances:

        >>> a = ivy.array([5, 10, 64])
        >>> b = ivy.Container(a = ivy.array([0, 1, 2]), b = ivy.array([3]))
        >>> y = a >> b
        >>> print(y)
        {
            a: ivy.array([5, 5, 16]),
            b: ivy.array([0, 1, 8])
        }
        """
        return ivy.bitwise_right_shift(self._data, other)

    @_native_wrapper
    def __rrshift__(self, other):
        """
        ivy.Array reverse special method variant of ivy.bitwise_right_shift. 
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_right_shift also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer data type.
        other
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). 
            Should have an integer data type. Each element must be greater than or equal
            to ``0``.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have
            a data type determined by :ref:`type-promotion`.

        Examples
        --------
        
        >>> a = 32
        >>> b = ivy.array([0, 1, 2])
        >>> y = a >> b
        >>> print(y)
        ivy.array([32, 16,  8])
        """
        return ivy.bitwise_right_shift(other, self._data)

    # noinspection PyDefaultArgument
    @_native_wrapper
    def __deepcopy__(self, memodict={}):
        try:
            return to_ivy(self._data.__deepcopy__(memodict))
        except AttributeError:
            # ToDo: try and find more elegant solution to jax inability to
            #  deepcopy device arrays
            if ivy.current_backend_str() == "jax":
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

    def __repr__(self):
        return super().__repr__().replace("array", "variable")
