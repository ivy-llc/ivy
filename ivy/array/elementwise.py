# global
import abc
from typing import Optional, Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyUnresolvedReferences
class ArrayWithElementwise(abc.ABC):
    def abs(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.

        Examples
        --------
        Using :code:`ivy.Array` instance method:

        >>> x = ivy.array([2.6, -6.6, 1.6, -0])
        >>> y = x.abs()
        >>> print(y)
        ivy.array([ 2.6, 6.6, 1.6, 0.])
        """
        return ivy.abs(self, out=out)

    def acosh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.acosh(self._data, out=out)

    def acos(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.acos(self._data, out=out)

    def add(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
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
        return ivy.asin(self._data, out=out)

    def asinh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.asinh. This method simply wraps the
        function, and so the docstring for ivy.asinh also applies to this method
        with minimal changes.

        Examples
        --------
        Using :code:`ivy.Array` instance method:

        >>> x = ivy.array([-1., 0., 3.])
        >>> y = x.asinh()
        >>> print(y)
        ivy.array([-0.881,  0.   ,  1.82 ])
        """
        return ivy.asinh(self._data, out=out)

    def atan(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.atan(self._data, out=out)

    def atan2(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.atan2(self._data, x2, out=out)

    def atanh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.atanh(self._data, out=out)

    def bitwise_and(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_and(self._data, x2, out=out)

    def bitwise_left_shift(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_left_shift(self._data, x2, out=out)

    def bitwise_invert(
        self: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.bitwise_invert(self._data, out=out)

    def bitwise_or(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_or(self._data, x2, out=out)

    def bitwise_right_shift(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_right_shift(self._data, x2, out=out)

    def bitwise_xor(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_xor(self._data, x2, out=out)

    def ceil(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.ceil. This method simply wraps the
        function, and so the docstring for ivy.ceil also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([5.5, -2.5, 1.5, -0])
        >>> y = x.ceil()
        >>> print(y)
        ivy.array([ 6., -2.,  2.,  0.])
        """
        return ivy.ceil(self._data, out=out)

    def cos(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.cos(self._data, out=out)

    def cosh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.cosh(self._data, out=out)

    def divide(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.divide(self._data, x2, out=out)

    def equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.equal(self._data, x2, out=out)

    def exp(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.exp(self._data, out=out)

    def expm1(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.expm1(self._data, out=out)

    def floor(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.floor. This method simply wraps the
        function, and so the docstring for ivy.floor also applies to this method
        with minimal changes.

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
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.floor_divide(self._data, x2, out=out)

    def greater(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.greater(self._data, x2, out=out)

    def greater_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.greater_equal(self._data, x2, out=out)

    def isfinite(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isfinite(self._data, out=out)

    def isinf(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isinf(self._data, out=out)

    def isnan(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isnan(self._data, out=out)

    def less(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.less(self._data, x2, out=out)

    def less_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.less_equal(self._data, x2, out=out)

    def log(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log(self._data, out=out)

    def log1p(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log1p(self._data, out=out)

    def log2(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log2(self._data, out=out)

    def log10(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log10(self._data, out=out)

    def logaddexp(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.logaddexp(self._data, x2, out=out)

    def logical_and(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.logical_and(self._data, x2, out=out)

    def logical_not(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.logical_not(self._data, out=out)

    def logical_or(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.logical_or(self._data, x2, out=out)

    def logical_xor(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.logical_xor(self._data, x2, out=out)

    def multiply(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.multiply(self._data, x2, out=out)

    def negative(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.negative(self._data, out=out)

    def not_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.not_equal(self._data, x2, out=out)

    def positive(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.positive(self._data, out=out)

    def pow(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.pow(self._data, x2, out=out)

    def remainder(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.remainder(self._data, x2, out=out)

    def round(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.round(self._data, out=out)

    def sign(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sign. This method simply wraps the
        function, and so the docstring for ivy.sign also applies to this method
        with minimal changes.

        Examples
        --------
        Using :code:`ivy.Array` instance method:

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

        Examples
        --------
        With :code:`ivy.Array` input:

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
        return ivy.square(self._data, out=out)

    def sqrt(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sqrt(self._data, out=out)

    def subtract(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
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
            an array containing the tangent of each element in ``self``. The return must
            have a floating-point data type determined by :ref:`type-promotion`.

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

        Examples
        --------
        >>> x = ivy.array([0., 1., 2.])
        >>> y = x.tanh()
        >>> print(y)
        ivy.array([0., 0.762, 0.964])
        """
        return ivy.tanh(self._data, out=out)

    def trunc(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.trunc(self._data, out=out)

    def erf(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.erf(self._data, out=out)
