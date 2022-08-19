# global
import abc
from typing import Optional, Union

# local
import ivy


# noinspection PyUnresolvedReferences
class ArrayWithElementwise(abc.ABC):
    def abs(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the absolute value of each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.array([2.6, -6.6, 1.6, -0])
        >>> y = x.abs()
        >>> print(y)
        ivy.array([ 2.6, 6.6, 1.6, 0.])
        """
        return ivy.abs(self, out=out)

    def acosh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.acosh. This method simply wraps the
        function, and so the docstring for ivy.acosh also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent the area of a hyperbolic sector.
            Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse hyperbolic cosine
            of each element in ``self``.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.array([2., 10.0, 1.0])
        >>> y = x.acosh()
        >>> print(y)
        ivy.array([1.32, 2.99, 0.  ])

        """
        return ivy.acosh(self._data, out=out)

    def acos(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.acos. This method simply wraps the
        function, and so the docstring for ivy.acos also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse cosine of each element in ``self``.
            The  returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.array([1.0, 0.0, -0.9])
        >>> y = x.acos()
        >>> print(y)
        ivy.array([0.  , 1.57, 2.69])

        """
        return ivy.acos(self._data, out=out)

    def add(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.add. This method simply wraps the
        function, and so the docstring for ivy.add also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([4, 5, 6])
        >>> z = x.add(y)
        >>> print(z)
        ivy.array([5, 7, 9])
        """
        return ivy.add(self._data, x2, out=out)

    def asin(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.asin. This method simply wraps the
        function, and so the docstring for ivy.asin also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse sine of each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        Using :code:`ivy.Array` instance method:

        >>> x = ivy.array([-1., 1., 4., 0.8])
        >>> y = x.asin()
        >>> print(y)
        ivy.array([-1.57, 1.57, nan, 0.927])

        >>> x = ivy.array([-3., -0.9, 1.5, 2.8])
        >>> y = ivy.zeros(4)
        >>> x.asin(out=y)
        >>> print(y)
        ivy.array([nan, -1.12, nan, nan])
        """
        return ivy.asin(self._data, out=out)

    def asinh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.asinh. This method simply wraps the
        function, and so the docstring for ivy.asinh also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent the area of a hyperbolic sector.
            Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse hyperbolic sine of each element in ``self``.
            The returned array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([-1., 0., 3.])
        >>> y = x.asinh()
        >>> print(y)
        ivy.array([-0.881,  0.   ,  1.82 ])
        """
        return ivy.asinh(self._data, out=out)

    def atan(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.atan. This method simply wraps the
        function, and so the docstring for ivy.atan also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse tangent of each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.array([1.0, 0.5, -0.5])
        >>> y = x.atan()
        >>> print(y)
        ivy.array([ 0.785,  0.464, -0.464])

        """
        return ivy.atan(self._data, out=out)

    def atan2(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.atan2. This method simply wraps the
        function, and so the docstring for ivy.atan2 also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array corresponding to the y-coordinates.
            Should have a real-valued floating-point data type.
        x2
            second input array corresponding to the x-coordinates.
            Must be compatible with ``self``(see :ref:`broadcasting`).
            Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse tangent of the quotient ``self/x2``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([1.0, 0.5, 0.0, -0.5, 0.0])
        >>> y = ivy.array([1.0, 2.0, -1.5, 0, 1.0])
        >>> z = x.atan2(y)
        >>> print(z)
        ivy.array([ 0.785,  0.245,  3.14 , -1.57 ,  0.   ])
        """
        return ivy.atan2(self._data, x2, out=out)

    def atanh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.atanh. This method simply wraps the
        function, and so the docstring for ivy.atanh also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent the area of a hyperbolic sector.
            Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse hyperbolic tangent of each element
            in ``self``. The returned array must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0.0, 0.5, -0.9])
        >>> y = x.atanh()
        >>> print(y)
        ivy.array([ 0.   ,  0.549, -1.47 ])

        """
        return ivy.atanh(self._data, out=out)

    def bitwise_and(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.bitwise_and.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_and also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.
        """
        return ivy.bitwise_and(self._data, x2, out=out)

    def bitwise_left_shift(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.bitwise_left_shift.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_left_shift also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.
        """
        return ivy.bitwise_left_shift(self._data, x2, out=out)

    def bitwise_invert(
        self: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.bitwise_invert.
        This method simply wraps the function, and so the docstring
        for ivy.bitiwse_invert also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have an integer or boolean data type.

        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have the same data type as ``self``.
        """
        return ivy.bitwise_invert(self._data, out=out)

    def bitwise_or(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.bitwise_or. This method simply
        wraps the function, and so the docstring for ivy.bitwise_or also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``

        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([4, 5, 6])
        >>> z = x.bitwise_or(y)
        >>> print(z)
        ivy.array([5, 7, 7])
        """
        return ivy.bitwise_or(self._data, x2, out=out)

    def bitwise_right_shift(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.bitwise_right_shift.
        This method simply wraps the function, and so the docstring
        for ivy.bitwise_right_shift also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> a = ivy.array([[2, 3, 4], [5, 10, 64]])
        >>> b = ivy.array([0, 1, 2])
        >>> y = a.bitwise_right_shift(b)
        >>> print(y)
        ivy.array([[ 2,  1,  1],
                    [ 5,  5, 16]])
        """
        return ivy.bitwise_right_shift(self._data, x2, out=out)

    def bitwise_xor(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.bitwise_xor.
        This method simply wraps the function, and so the docstring
        for ivy.bitwise_xor also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.
        """
        return ivy.bitwise_xor(self._data, x2, out=out)

    def ceil(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.ceil.
        This method simply wraps the function, and so the docstring for
        ivy.ceil also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the rounded result for each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.array([5.5, -2.5, 1.5, -0])
        >>> y = x.ceil()
        >>> print(y)
        ivy.array([ 6., -2.,  2.,  0.])
        """
        return ivy.ceil(self._data, out=out)

    def cos(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cos. This method simply wraps the
        function, and so the docstring for ivy.cos also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are each expressed in radians. Should have a
            floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the cosine of each element in ``self``. The returned
            array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Array` input:

        >>> x = ivy.array([1., 0., 2.,])
        >>> y = x.cos()
        >>> print(y)
        ivy.array([0.54, 1., -0.416])

        >>> x = ivy.array([-3., 0., 3.])
        >>> y = ivy.zeros(3)
        >>> ivy.cos(x, out=y)
        >>> print(y)
        ivy.array([-0.99,  1.  , -0.99])
        """
        return ivy.cos(self._data, out=out)

    def cosh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cosh. This method simply wraps
        the function, and so the docstring for ivy.cosh also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent a hyperbolic angle.
            Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hyperbolic cosine of each element in ``self``.
            The returned array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([1., 2., 3.])
        >>> print(x.cosh())
            ivy.array([1.54, 3.76, 10.1])

        >>> x = ivy.array([0.23, 3., -1.2])
        >>> y = ivy.zeros(3)
        >>> print(x.cosh(out=y))
            ivy.array([1.03, 10.1, 1.81])
        """
        return ivy.cosh(self._data, out=out)

    def divide(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.divide. This method simply
        wraps the function, and so the docstring for ivy.divide also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array. Should have a real-valued data type.
        x2
            divisor input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Array` inputs:

        >>> x1 = ivy.array([2., 7., 9.])
        >>> x2 = ivy.array([2., 2., 2.])
        >>> y = x1.divide(x2)
        >>> print(y)
        ivy.array([1., 3.5, 4.5])

        With mixed :code:`ivy.Array` and `ivy.NativeArray` inputs:

        >>> x1 = ivy.array([2., 7., 9.])
        >>> x2 = ivy.native_array([2., 2., 2.])
        >>> y = x1.divide(x2)
        >>> print(y)
        ivy.array([1., 3.5, 4.5])
        """
        return ivy.divide(self._data, x2, out=out)

    def equal(
        self: ivy.Array,
        x2: Union[float, ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.equal.
        This method simply wraps the function, and so the docstring for
        ivy.equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            May have any data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        With :code:`ivy.Array` inputs:

        >>> x1 = ivy.array([2., 7., 9.])
        >>> x2 = ivy.array([1., 7., 9.])
        >>> y = x1.equal(x2)
        >>> print(y)
        ivy.array([False, True, True])

        With mixed :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

        >>> x1 = ivy.array([2.5, 7.3, 9.375])
        >>> x2 = ivy.native_array([2.5, 2.9, 9.375])
        >>> y = x1.equal(x2)
        >>> print(y)
        ivy.array([True, False,  True])

        With mixed :code:`ivy.Array` and `float` inputs:

        >>> x1 = ivy.array([2.5, 7.3, 9.375])
        >>> x2 = 7.3
        >>> y = x1.equal(x2)
        >>> print(y)
        ivy.array([False, True, False])

        With mixed :code:`ivy.Container` and :code:`ivy.Array` inputs:

        >>> x1 = ivy.array([3., 1., 0.9])
        >>> x2 = ivy.Container(a=ivy.array([12., 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
        >>> y = x1.equal(x2)
        >>> print(y)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([True, True, True])
        }
        """
        return ivy.equal(self._data, x2, out=out)

    def exp(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.exp. This method simply
        wraps the function, and so the docstring for ivy.exp also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated exponential function result for
            each element in ``self``. The returned array must have a floating-point
            data type determined by :ref:`type-promotion`.
        """
        return ivy.exp(self._data, out=out)

    def expm1(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.expm1. This method simply wraps the
        function, and so the docstring for ivy.expm1 also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([5.5, -2.5, 1.5, -0])
        >>> y = x.expm1()
        >>> print(y)
        ivy.array([244.   ,  -0.918,   3.48 ,   0.   ])

        >>> y = ivy.array([0., 0.])
        >>> x = ivy.array([5., 0.])
        >>> _ = x.expm1(out=y)
        >>> print(y)
        ivy.array([147.,   0.])
        """
        return ivy.expm1(self._data, out=out)

    def floor(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.floor. This method simply wraps
        the function, and so the docstring for ivy.floor also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the rounded result for each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.array([5.5, -2.5, 1.5, -0])
        >>> y = x.floor()
        >>> print(y)
        ivy.array([ 5., -3.,  1.,  0.])
        """
        return ivy.floor(self._data, out=out)

    def floor_divide(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.floor_divide.
        This method simply wraps the function, and so the docstring for ivy.floor_divide
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array. Should have a real-valued data type.
        x2
            divisor input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Array` inputs:

        >>> x1 = ivy.array([13., 7., 8.])
        >>> x2 = ivy.array([3., 2., 7.])
        >>> y = x1.floor_divide(x2)
        >>> print(y)
        ivy.array([4., 3., 1.])

        With mixed :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

        >>> x1 = ivy.array([13., 7., 8.])
        >>> x2 = ivy.native_array([3., 2., 7.])
        >>> y = x1.floor_divide(x2)
        >>> print(y)
        ivy.array([4., 3., 1.])
        """
        return ivy.floor_divide(self._data, x2, out=out)

    def greater(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.greater.
        This method simply wraps the function, and so the docstring for
        ivy.greater also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must
            have a data type of ``bool``.

        Examples
        --------
        >>> x1 = ivy.array([2., 5., 15.])
        >>> x2 = ivy.array([3., 2., 4.])
        >>> y = x1.greater(x2)
        >>> print(y)
        ivy.array([False,  True,  True])

        """
        return ivy.greater(self._data, x2, out=out)

    def greater_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.greater_equal.
        This method simply wraps the function, and so the docstring for
        ivy.greater_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.
        """
        return ivy.greater_equal(self._data, x2, out=out)

    def isfinite(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.isfinite.
        This method simply wraps the function, and so the docstring
        for ivy.isfinite also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing test results. An element ``out_i`` is ``True``
            if ``self_i`` is finite and ``False`` otherwise.
            The returned array must have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.array([0, ivy.nan, -ivy.inf, float('inf')])
        >>> y = x.isfinite()
        >>> print(y)
        ivy.array([ True, False, False, False])
        """
        return ivy.isfinite(self._data, out=out)

    def isinf(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.isinf. This method simply wraps
        the function, and so the docstring for ivy.isinf also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing test results. An element ``out_i`` is ``True``
            if ``self_i`` is either positive or negative infinity and ``False``
            otherwise. The returned array must have a data type of ``bool``.
        """
        return ivy.isinf(self._data, out=out)

    def isnan(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.isnan. This method simply wraps
        the function, and so the docstring for ivy.isnan also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing test results. An element ``out_i`` is ``True``
            if ``self_i`` is ``NaN`` and ``False`` otherwise.
            The returned array should have a data type of ``bool``.
        """
        return ivy.isnan(self._data, out=out)

    def less(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.less. This method simply wraps
        the function, and so the docstring for ivy.less also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        >>> x1 = ivy.array([2., 5., 15.])
        >>> x2 = ivy.array([3., 2., 4.])
        >>> y = x1.less(x2)
        >>> print(y)
        ivy.array([ True, False, False])

        """
        return ivy.less(self._data, x2, out=out)

    def less_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.less_equal.
        This method simply wraps the function, and so the docstring
        for ivy.less_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        With :code:'ivy.Array' inputs:

        >>> x1 = ivy.array([1, 2, 3])
        >>> x2 = ivy.array([2, 2, 1])
        >>> y = x1.less_equal(x2)
        >>> print(y)
        ivy.array([True, True, False])

        With mixed :code:'ivy.Array' and :code:'ivy.NativeArray' inputs:

        >>> x1 = ivy.array([2.5, 3.3, 9.24])
        >>> x2 = ivy.native_array([2.5, 1.1, 9.24])
        >>> y = x1.less_equal(x2)
        >>> print(y)
        ivy.array([True, False, True])

        With mixed :code:'ivy.Container' and :code:'ivy.Array' inputs:

        >>> x1 = ivy.array([3., 1., 0.8])
        >>> x2 = ivy.Container(a=ivy.array([2., 1., 0.7]), b=ivy.array([3., 0.6, 1.2]))
        >>> y = x1.less_equal(x2)
        >>> print(y)
        {
            a: ivy.array([False, True, False]),
            b: ivy.array([True, False, True])
        }
        """
        return ivy.less_equal(self._data, x2, out=out)

    def log(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.log. This method simply wraps the
        function, and so the docstring for ivy.log also applies to this method
        with minimal changes.


        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.
        """
        return ivy.log(self._data, out=out)

    def log1p(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.log1p.
        This method simply wraps the function, and so the docstring
        for ivy.log1p also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([1 , 2 ,3 ])
        >>> y = x.log1p()
        >>> print(y)
        ivy.array([0.693, 1.1  , 1.39 ])

        >>> x = ivy.array([0.1 , .001 ])
        >>> x.log1p( out = x)
        >>> print(x)
        ivy.array([0.0953, 0.001 ])

        """
        return ivy.log1p(self._data, out=out)

    def log2(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.log2.
        This method simply wraps the function, and so the docstring for
        ivy.log2 also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated base ``2`` logarithm for each element
            in ``self``. The returned array must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.
        """
        return ivy.log2(self._data, out=out)

    def log10(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.log10. This method simply wraps the
        function, and so the docstring for ivy.log10 also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated base ``10`` logarithm for each element
            in ``self``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :code:`ivy.Array` instance method:

        >>> x = ivy.array([4.0, 1, -0.0, -5.0])
        >>> y = x.log10()
        >>> print(y)
        ivy.array([0.602, 0., -inf, nan])

        >>> x = ivy.array([float('nan'), -5.0, -0.0, 1.0, 5.0, float('+inf')])
        >>> y = x.log10()
        >>> print(y)
        ivy.array([nan, nan, -inf, 0., 0.699, inf])

        >>> x = ivy.array([[float('nan'), 1, 5.0, float('+inf')],\
                           [+0, -1.0, -5, float('-inf')]])
        >>> y = x.log10()
        >>> print(y)
        ivy.array([[nan, 0., 0.699, inf],
                   [-inf, nan, nan, nan]])
        """
        return ivy.log10(self._data, out=out)

    def logaddexp(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.logaddexp.
        This method simply wraps the function, and so the docstring for
        ivy.logaddexp also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a real-valued floating-point data
            type determined by :ref:`type-promotion`.
        """
        return ivy.logaddexp(self._data, x2, out=out)

    def logical_and(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.logical_and.
        This method simply wraps the function, and so the docstring for
        ivy.logical_and also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        Using 'ivy.Array' instance:

        >>> x = ivy.array([True, False, True, False])
        >>> y = ivy.array([True, True, False, False])
        >>> z = x.logical_and(y)
        >>> print(z)
        ivy.array([True, False, False, False])
        """
        return ivy.logical_and(self._data, x2, out=out)

    def logical_not(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.logical_not.
        This method simply wraps the function, and so the docstring
        for ivy.logical_not also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a boolean data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type of ``bool``.
        """
        return ivy.logical_not(self._data, out=out)

    def logical_or(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.logical_or.
        This method simply wraps the function, and so the docstring for
        ivy.logical_or also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        This function conforms to the `Array API Standard
        <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
        `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.logical_or.html>`_ # noqa
        in the standard.

        Both the description and the type hints above assumes an array input for simplicity,
        but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
        instances in place of any of the arguments.

        Examples
        --------
        Using :code:`ivy.Array` instance method:

        >>> x = ivy.array([False, 3, 0])
        >>> y = ivy.array([2, True, False])
        >>> z = x.logical_or(y)
        >>> print(z)
        ivy.array([ True,  True, False])
        """
        return ivy.logical_or(self._data, x2, out=out)

    def logical_xor(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.logical_xor.
        This method simply wraps the function, and so the docstring
        for ivy.logical_xor also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.array([True, False, True, False])
        >>> y = ivy.array([True, True, False, False])
        >>> z = x.logical_xor(y)
        >>> print(z)
        ivy.array([False,  True,  True, False])
        """
        return ivy.logical_xor(self._data, x2, out=out)

    def multiply(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.multiply.
        This method simply wraps the function, and so the docstring for ivy.multiply
         also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise products. The returned array
            must have a data type determined by :ref:`type-promotion`.
        """
        return ivy.multiply(self._data, x2, out=out)

    def negative(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.negative.
        This method simply wraps the function, and so the docstring
        for ivy.negative also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``.
            The returned array must have the same data type as ``self``.
        """
        return ivy.negative(self._data, out=out)

    def not_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.not_equal.
        This method simply wraps the function, and so the docstring
        for ivy.not_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned
            array must have a data type of ``bool``.
        """
        return ivy.not_equal(self._data, x2, out=out)

    def positive(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.positive.
        This method simply wraps the function, and so the docstring
        for ivy.positive also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``.
            The returned array must have the same data type as ``self``.
        """
        return ivy.positive(self._data, out=out)

    def pow(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.pow. This method simply wraps the
        function, and so the docstring for ivy.pow also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array whose elements correspond to the exponentiation base.
            Should have a real-valued data type.
        x2
            second input array whose elements correspond to the exponentiation
            exponent. Must be compatible with ``self`` (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.
        """
        return ivy.pow(self._data, x2, out=out)

    def remainder(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.remainder.
        This method simply wraps the function, and so the docstring
        for ivy.remainder also applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array. Should have a real-valued data type.
        x2
            divisor input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            Each element-wise result must have the same sign as the respective
            element ``x2_i``. The returned array must have a data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        With :code: `ivy.Array` inputs:

        >>> x1 = ivy.array([2., 5., 15.])
        >>> x2 = ivy.array([3., 2., 4.])
        >>> y = x1.remainder(x2)
        >>> print(y)
        ivy.array([2., 1., 3.])

        With mixed :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

        >>> x1 = ivy.array([11., 4., 18.])
        >>> x2 = ivy.native_array([2., 5., 8.])
        >>> y = x1.remainder(x2)
        >>> print(y)
        ivy.array([1., 4., 2.])
        """
        return ivy.remainder(self._data, x2, out=out)

    def round(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.round. This method simply wraps the
        function, and so the docstring for ivy.round also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the rounded result for each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        Using :code:`ivy.Array` instance method:

        >>> x = ivy.array([6.3, -8.1, 0.5, -4.2, 6.8])
        >>> y = x.round()
        >>> print(y)
        ivy.array([ 6., -8.,  0., -4.,  7.])

        >>> x = ivy.array([-94.2, 256.0, 0.0001, -5.5, 36.6])
        >>> y = x.round()
        >>> print(y)
        ivy.array([-94., 256., 0., -6., 37.])

        >>> x = ivy.array([0.23, 3., -1.2])
        >>> y = ivy.zeros(3)
        >>> x.round(out=y)
        >>> print(y)
        ivy.array([ 0.,  3., -1.])

        >>> x = ivy.array([[ -1., -67.,  0.,  15.5,  1.], [3, -45, 24.7, -678.5, 32.8]])
        >>> y = x.round()
        >>> print(y)
        ivy.array([[-1., -67., 0., 16., 1.],
        [3., -45., 25., -678., 33.]])
        """
        return ivy.round(self._data, out=out)

    def sign(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sign. This method simply wraps the
        function, and so the docstring for ivy.sign also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.array([5.7, -7.1, 0, -0, 6.8])
        >>> y = x.sign()
        >>> print(y)
        ivy.array([ 1., -1.,  0.,  0.,  1.])

        >>> x = ivy.array([-94.2, 256.0, 0.0001, -0.0001, 36.6])
        >>> y = x.sign()
        >>> print(y)
        ivy.array([-1.,  1.,  1., -1.,  1.])

        >>> x = ivy.array([[ -1., -67.,  0.,  15.5,  1.], [3, -45, 24.7, -678.5, 32.8]])
        >>> y = x.sign()
        >>> print(y)
        ivy.array([[-1., -1.,  0.,  1.,  1.],
        [ 1., -1.,  1., -1.,  1.]])
        """
        return ivy.sign(self._data, out=out)

    def sin(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sin. This method simply wraps the
        function, and so the docstring for ivy.sin also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are each expressed in radians. Should have a
            floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the sine of each element in ``self``. The returned
            array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2., 3.])
        >>> y = x.sin()
        >>> print(y)
        ivy.array([0., 0.841, 0.909, 0.141])
        """
        return ivy.sin(self._data, out=out)

    def sinh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sinh. This method simply wraps the
        function, and so the docstring for ivy.sinh also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent a hyperbolic angle.
            Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hyperbolic sine of each element in ``self``. The
            returned array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([1., 2., 3.])
        >>> print(x.sinh())
            ivy.array([1.18, 3.63, 10.])

        >>> x = ivy.array([0.23, 3., -1.2])
        >>> y = ivy.zeros(3)
        >>> print(x.sinh(out=y))
            ivy.array([0.232, 10., -1.51])
        """
        return ivy.sinh(self._data, out=out)

    def square(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.square.
        This method simply wraps the function, and so the docstring
        for ivy.square also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the square of each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.
        """
        return ivy.square(self._data, out=out)

    def sqrt(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sqrt. This method simply wraps the
        function, and so the docstring for ivy.sqrt also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the square root of each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.
        """
        return ivy.sqrt(self._data, out=out)

    def subtract(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.subtract.
        This method simply wraps the function, and so the docstring
        for ivy.subtract also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise differences. The returned array
            must have a data type determined by :ref:`type-promotion`.
        """
        return ivy.subtract(self._data, x2, out=out)

    def tan(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tan. This method simply wraps the
        function, and so the docstring for ivy.tan also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are expressed in radians. Should have a
            floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the tangent of each element in ``self``.
            The return must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2.])
        >>> y = x.tan()
        >>> print(y)
        ivy.array([0., 1.56, -2.19])
        """
        return ivy.tan(self._data, out=out)

    def tanh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tanh. This method simply wraps the
        function, and so the docstring for ivy.tanh also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent a hyperbolic angle.
            Should have a real-valued floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hyperbolic tangent of each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2.])
        >>> y = x.tanh()
        >>> print(y)
        ivy.array([0., 0.762, 0.964])
        """
        return ivy.tanh(self._data, out=out)

    def trunc(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.trunc. This method simply wraps the
        function, and so the docstring for ivy.trunc also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the rounded result for each element in ``self``.
            The returned array must have the same data type as ``self``.
        """
        return ivy.trunc(self._data, out=out)

    def erf(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.erf. This method simply wraps the
        function, and so the docstring for ivy.erf also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array to compute exponential for.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the Gauss error of ``self``.
        """
        return ivy.erf(self._data, out=out)
