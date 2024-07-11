# global
import abc
from typing import Tuple, Optional, List, Union

# local
import ivy

Finfo = None
Iinfo = None


class _ArrayWithDataTypes(abc.ABC):
    def astype(
        self: ivy.Array,
        dtype: ivy.Dtype,
        /,
        *,
        copy: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Copy an array to a specified data type irrespective of :ref:`type-
        promotion` rules.

        .. note::
        Casting floating-point ``NaN`` and ``infinity`` values to integral data types
        is not specified and is implementation-dependent.

        .. note::
        When casting a boolean input array to a numeric data type, a value of ``True``
        must cast to a numeric value equal to ``1``, and a value of ``False`` must cast
        to a numeric value equal to ``0``.

        When casting a numeric input array to ``bool``, a value of ``0`` must cast to
        ``False``, and a non-zero value must cast to ``True``.

        Parameters
        ----------
        self
            array to cast.
        dtype
            desired data type.
        copy
            specifies whether to copy an array when the specified ``dtype`` matches
            the data type of the input array ``x``. If ``True``, a newly allocated
            array must always be returned. If ``False`` and the specified ``dtype``
            matches the data type of the input array, the input array must be returned;
            otherwise, a newly allocated must be returned. Default: ``True``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array having the specified data type. The returned array must have
            the same shape as ``x``.

        Examples
        --------
        Using :class:`ivy.Array` instance method:

        >>> x = ivy.array([[-1, -2], [0, 2]])
        >>> print(x.astype(ivy.float64))
        ivy.array([[-1., -2.],  [0.,  2.]])
        """
        return ivy.astype(self._data, dtype, copy=copy, out=out)

    def broadcast_arrays(
        self: ivy.Array, *arrays: Union[ivy.Array, ivy.NativeArray]
    ) -> List[ivy.Array]:
        """`ivy.Array` instance method variant of `ivy.broadcast_arrays`. This
        method simply wraps the function, and so the docstring for
        `ivy.broadcast_arrays` also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            An input array to be broadcasted against other input arrays.
        arrays
            an arbitrary number of arrays to-be broadcasted.
            Each array must have the same shape.
            Each array must have the same dtype as its
            corresponding input array.

        Returns
        -------
        ret
            A list containing broadcasted arrays of type `ivy.Array`

        Examples
        --------
        With :class:`ivy.Array` inputs:

        >>> x1 = ivy.array([1, 2])
        >>> x2 = ivy.array([0.2, 0.])
        >>> x3 = ivy.zeros(2)
        >>> y = x1.broadcast_arrays(x2, x3)
        >>> print(y)
        [ivy.array([1, 2]), ivy.array([0.2, 0. ]), ivy.array([0., 0.])]

        With mixed :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

        >>> x1 = ivy.array([-1., 3.4])
        >>> x2 = ivy.native_array([2.4, 5.1])
        >>> y = x1.broadcast_arrays(x2)
        >>> print(y)
        [ivy.array([-1., 3.4]), ivy.array([2.4, 5.1])]
        """
        return ivy.broadcast_arrays(self._data, *arrays)

    def broadcast_to(
        self: ivy.Array, /, shape: Tuple[int, ...], *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """`ivy.Array` instance method variant of `ivy.broadcast_to`. This
        method simply wraps the function, and so the docstring for
        `ivy.broadcast_to` also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array to be broadcasted.
        shape
            desired shape to be broadcasted to.
        out
            Optional array to store the broadcasted array.

        Returns
        -------
        ret
            Returns the broadcasted array of shape 'shape'

        Examples
        --------
        With :class:`ivy.Array` instance method:

        >>> x = ivy.array([1, 2, 3])
        >>> y = x.broadcast_to((3,3))
        >>> print(y)
        ivy.array([[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]])
        """
        return ivy.broadcast_to(self._data, shape=shape, out=out)

    def can_cast(self: ivy.Array, to: ivy.Dtype) -> bool:
        """`ivy.Array` instance method variant of `ivy.can_cast`. This method
        simply wraps the function, and so the docstring for `ivy.can_cast` also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array from which to cast.
        to
            desired data type.

        Returns
        -------
        ret
            ``True`` if the cast can occur according to :ref:`type-promotion` rules;
            otherwise, ``False``.

        Examples
        --------
        >>> x = ivy.array([1., 2., 3.])
        >>> print(x.dtype)
        float32

        >>> x = ivy.array([4., 5., 6.])
        >>> print(x.can_cast(ivy.float64))
        True
        """
        return ivy.can_cast(self._data, to)

    def dtype(
        self: ivy.Array, as_native: bool = False
    ) -> Union[ivy.Dtype, ivy.NativeDtype]:
        """`ivy.Array` instance method variant of `ivy.dtype`. This method
        helps to get the data type of the array.

        Parameters
        ----------
        self
            The input array.
        as_native
            Whether to return the native data type of the array.
            If True, returns the native data type. Default is False.

        Returns
        -------
        ret
            The data type of the array. If as_native is True,
            returns the native data type.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = x.dtype()
        >>> print(y)
        int32

        >>> x= ivy.array([1.0, 2.0, 3.0], dtype=ivy.float64)
        >>> y = x.dtype(as_native=True)
        >>> print(y)
        float64
        """
        return ivy.dtype(self._data, as_native=as_native)

    def finfo(self: ivy.Array, /) -> Finfo:
        """Array instance method variant of `ivy.finfo`.

        Parameters
        ----------
        self
            input array.

        Returns
        -------
        ret
            An instance of the `Finfo` class, containing information
            about the floating point data type of the input array.

        Example
        -------
        >>> x = ivy.array([0.7,8.4,3.14], dtype=ivy.float32)
        >>> print(x.finfo())
        finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)
        """
        return ivy.finfo(self._data)

    def iinfo(self: ivy.Array, /) -> Iinfo:
        """`ivy.Array` instance method variant of `ivy.iinfo`. This method
        simply wraps the function, and so the docstring for `ivy.iinfo` also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.

        Returns
        -------
        ret
            An instance of the `Iinfo` class, containing information
            about the integer data type of the input array.

        Examples
        --------
        >>> x = ivy.array([-119,122,14], dtype=ivy.int8))
        >>> x.iinfo()
        iinfo(min=-128, max=127, dtype=int8)

        >>> x = ivy.array([-12,54,1,9,-1220], dtype=ivy.int16))
        >>> x.iinfo()
        iinfo(min=-32768, max=32767, dtype=int16)
        """
        return ivy.iinfo(self._data)

    def is_bool_dtype(self: ivy.Array) -> bool:
        return ivy.is_bool_dtype(self._data)

    def is_float_dtype(self: ivy.Array) -> bool:
        """`ivy.Array` instance method variant of `ivy.is_float_dtype`. This
        method simply checks to see if the array is of type `float`.

        Parameters
        ----------
        self
            Input array from which to check for float dtype.

        Returns
        -------
        ret
            Boolean value of whether the array is of type `float`.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3], dtype=ivy.int8)
        >>> print(x.is_float_dtype())
        False

        >>> x = ivy.array([2.3, 4.5, 6.8], dtype=ivy.float32)
        >>> print( x.is_float_dtype())
        True
        """
        return ivy.is_float_dtype(self._data)

    def is_int_dtype(self: ivy.Array) -> bool:
        return ivy.is_int_dtype(self._data)

    def is_uint_dtype(self: ivy.Array) -> bool:
        return ivy.is_uint_dtype(self._data)

    def result_type(
        self: ivy.Array,
        *arrays_and_dtypes: Union[ivy.Array, ivy.NativeArray, ivy.Dtype],
    ) -> ivy.Dtype:
        """`ivy.Array` instance method variant of `ivy.result_type`. This
        method simply wraps the function, and so the docstring for
        `ivy.result_type` also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array from which to cast.
        arrays_and_dtypes
            an arbitrary number of input arrays and/or dtypes.

        Returns
        -------
        ret
            the dtype resulting from an operation involving the input arrays and dtypes.

        Examples
        --------
        >>> x = ivy.array([0, 1, 2])
        >>> print(x.dtype)
        int32

        >>> x.result_type(ivy.float64)
        <dtype:'float64'>
        """
        return ivy.result_type(self._data, *arrays_and_dtypes)
