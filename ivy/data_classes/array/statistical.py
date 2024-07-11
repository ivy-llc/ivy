# global
from typing import Optional, Union, Sequence
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class _ArrayWithStatistical(abc.ABC):
    def min(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        initial: Optional[Union[int, float, complex]] = None,
        where: Optional[ivy.Array] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Calculate the minimum value of the input array ``x``.

        Parameters
        ----------
        self
            Input array. Should have a real-valued data type.
        axis
            axis or axes along which minimum values must be computed.
            By default, the minimum value must be computed over the
            entire array. If a tuple of integers,minimum values must be
            computed over multiple axes. Default: ``None``.

        keepdims
            optional boolean, if ``True``, the reduced axes (dimensions)
            must be included in the result as singleton dimensions, and,
            accordingly, the result must be compatible with the input
            array (see :ref:`broadcasting`). Otherwise, if ``False``, the
            reduced axes (dimensions) must not be included in the
            result. Default: ``False``.
        initial
            The maximum value of an output element.
            Must be present to allow computation on empty slice.
        where
            Elements to compare for minimum
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            if the minimum value was computed over the entire array, a
            zero-dimensional array containing the minimum value; otherwise,
            a non-zero-dimensional array containing the minimum values.
            The returned array must have the same data type
            as ``x``.

        Examples
        --------
        With :code:`ivy.Array` input:

        >>> x = ivy.array([3., 4., 5.])
        >>> y = x.min()
        >>> print(y)
        ivy.array(3.)

        >>> x = ivy.array([[-1, 0, 1], [2, 3, 4]])
        >>> y = x.min(axis=1)
        >>> print(y)
        ivy.array([-1,  2])

        >>> x = ivy.array([0.1, 1.1, 2.1])
        >>> y = ivy.array(0.)
        >>> x.min(out=y)
        >>> print(y)
        ivy.array(0.1)
        """
        return ivy.min(
            self._data,
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            where=where,
            out=out,
        )

    def max(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.max. This method simply
        wraps the function, and so the docstring for ivy.max also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input array. Should have a numeric data type.
        axis
            axis or axes along which maximum values must be computed.
            By default, the maximum value must be computed over the
            entire array. If a tuple of integers, maximum values must
            be computed over multiple axes. Default: ``None``.
        keepdims
            if ``True``, the reduced axes (dimensions) must be included
            in the result as singleton dimensions, and, accordingly, the
            result must be compatible with the input array
            (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
            (dimensions) must not be included in the result. Default: ``False``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            if the maximum value was computed over the entire array,
            a zero-dimensional array containing the maximum value;
            otherwise, a non-zero-dimensional array
            containing the maximum values. The returned array must
            have the same data type
            as ``x``.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.array([1, 2, 3])
        >>> z = x.max()
        >>> print(z)
        ivy.array(3)

        >>> x = ivy.array([0, 1, 2])
        >>> z = ivy.array(0)
        >>> y = x.max(out=z)
        >>> print(z)
        ivy.array(2)

        >>> x = ivy.array([[0, 1, 2], [4, 6, 10]])
        >>> y = x.max(axis=0, keepdims=True)
        >>> print(y)
        ivy.array([[4, 6, 10]])
        """
        return ivy.max(self._data, axis=axis, keepdims=keepdims, out=out)

    def mean(
        self: ivy.Array,
        /,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.mean. This method simply
        wraps the function, and so the docstring for ivy.mean also applies to
        this method with minimal changes.

        **Special Cases**

        Let ``N`` equal the number of elements over which to compute the
        arithmetic mean.
        -   If ``N`` is ``0``, the arithmetic mean is ``NaN``.
        -   If ``x_i`` is ``NaN``, the arithmetic mean is ``NaN`` (i.e., ``NaN``
            values propagate).

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        axis
            axis or axes along which arithmetic means must be computed. By default,
            the mean must be computed over the entire array. If a Sequence of
            integers, arithmetic means must be computed over multiple axes.
            Default: ``None``.
        keepdims
            bool, if ``True``, the reduced axes (dimensions) must be included in the
            result as singleton dimensions, and, accordingly, the result must be
            compatible with the input array (see :ref:`broadcasting`). Otherwise,
            if ``False``, the reduced axes (dimensions) must not be included in
            the result. Default: ``False``.
        dtype
            the desired data type of returned tensor. If specified, the input tensor
            is casted to dtype before the operation is performed. This is useful for
            preventing data type overflows. Default: None.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            array, if the arithmetic mean was computed over the entire array, a
            zero-dimensional array containing the arithmetic mean; otherwise, a
            non-zero-dimensional array containing the arithmetic means.
            The returned array must have the same data type as ``x``.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.array([3., 4., 5.])
        >>> y = x.mean()
        >>> print(y)
        ivy.array(4.)

        >>> x = ivy.array([-1., 0., 1.])
        >>> y = ivy.mean(x)
        >>> print(y)
        ivy.array(0.)

        >>> x = ivy.array([0.1, 1.1, 2.1])
        >>> y = ivy.array(0.)
        >>> x.mean(out=y)
        >>> print(y)
        ivy.array(1.1)

        >>> x = ivy.array([1., 2., 3., 0., -1.])
        >>> y = ivy.array(0.)
        >>> ivy.mean(x, out=y)
        >>> print(y)
        ivy.array(1.)

        >>> x = ivy.array([[-0.5, 1., 2.], [0.0, 1.1, 2.2]])
        >>> y = ivy.zeros((1, 3))
        >>> x.mean(axis=0, keepdims=True, out=y)
        >>> print(y)
        ivy.array([[-0.25      ,  1.04999995,  2.0999999 ]])

        >>> x = ivy.array([[0., 1., 2.], [3., 4., 5.]])
        >>> y = ivy.array([0., 0.])
        >>> ivy.mean(x, axis=1, out=y)
        >>> print(y)
        ivy.array([1., 4.])
        """
        return ivy.mean(self._data, axis=axis, keepdims=keepdims, dtype=dtype, out=out)

    def var(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.var. This method simply
        wraps the function, and so the docstring for ivy.var also applies to
        this method with minimal changes.

        **Special Cases**

        Let N equal the number of elements over which to compute the variance.

        If N - correction is less than or equal to 0, the variance is NaN.

        If x_i is NaN, the variance is NaN (i.e., NaN values propagate).

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        axis
            axis or axes along which variances must be computed. By default, the
            variance must be computed over the entire array. If a tuple of integers,
            variances must be computed over multiple axes. Default: ``None``.
        correction
            degrees of freedom adjustment. Setting this parameter to a value other
            than 0 has the effect of adjusting the divisor during the calculation
            of the variance according to N-c where N corresponds to the total
            number of elements over which the variance is computed and c corresponds
            to the provided degrees of freedom adjustment. When computing the variance
            of a population, setting this parameter to 0 is the standard choice
            (i.e., the provided array contains data constituting an entire population).
            When computing the unbiased sample variance, setting this parameter to 1
            is the standard choice (i.e., the provided array contains data sampled
            from a larger population; this is commonly referred to as Bessel's
            correction). Default: ``0``.
        keepdims
            if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible
            with the input array (see Broadcasting). Otherwise, if False, the
            reduced axes (dimensions) must not be included in the result.
            Default: ``False``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            if the variance was computed over the entire array, a zero-dimensional array
            containing the variance; otherwise, a non-zero-dimensional array containing
            the variances. The returned array must have the same data type as x.

        Examples
        --------
        >>> x = ivy.array([[0.0, 1.0, 2.0],
        ...                [3.0, 4.0, 5.0],
        ...                [6.0, 7.0, 8.0]])
        >>> y = x.var()
        >>> print(y)
        ivy.array(6.6666665)

        >>> x = ivy.array([[0.0, 1.0, 2.0],
        ...                [3.0, 4.0, 5.0],
        ...                [6.0, 7.0, .08]])
        >>> y = x.var(axis=0)
        >>> print(y)
        ivy.array([6., 6., 4.1])

        >>> x = ivy.array([[0.0, 1.0, 2.0],
        ...                [3.0, 4.0, 5.0],
        ...                [6.0, 7.0, .08]])
        >>> y = ivy.array([0., 0., 0.])
        >>> x.var(axis=1, out=y)
        >>> print(y)
        ivy.array([0.667, 0.667, 9.33 ])
        """
        return ivy.var(
            self._data, axis=axis, correction=correction, keepdims=keepdims, out=out
        )

    def prod(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.array instance method variant of ivy.prod. This method simply
        wraps the function, and so the docstring for ivy.prod also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        axis
            axis or axes along which products must be computed. By default,
            the product must be computed over the entire array. If a
            tuple of integers, products must be computed over multiple
            axes. Default: ``None``.
        keepdims
            bool, if True, the reduced axes (dimensions) must be
            included in the result as singleton dimensions, and,
            accordingly, the result must be compatible with the
            input array (see Broadcasting). Otherwise, if False,
            the reduced axes (dimensions) must not be included in
            the result. Default: ``False``.
        dtype
            data type of the returned array.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            container, if the product was computed over the entire array,
            a zero-dimensional array containing the product;
            otherwise, a non-zero-dimensional array containing the products.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        With: class: `ivy.Array` input:

        >>> x = ivy.array([1, 2, 3])
        >>> z = x.prod()
        >>> print(z)
        ivy.array(6)

        >>> x = ivy.array([1, 0, 3])
        >>> z = x.prod()
        >>> print(z)
        ivy.array(0)

        >>> x = ivy.array([[3., 4., 5.]])
        >>> y = x.prod(axis=1)
        >>> print(y)
        ivy.array([60.])

        >>> x = ivy.array([2., 1.])
        >>> y = ivy.array(0.)
        >>> x.prod(out=y)
        >>> print(y)
        ivy.array(2.)

        >>> x = ivy.array([[-1., -2.], [3., 3.]])
        >>> y = x.prod(axis=1)
        >>> print(y)
        ivy.array([2., 9.])
        """
        return ivy.prod(self._data, axis=axis, keepdims=keepdims, dtype=dtype, out=out)

    def sum(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.sum(self, axis=axis, dtype=dtype, keepdims=keepdims, out=out)

    def std(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.array instance method variant of ivy.std. This method simply
        wraps the function, and so the docstring for ivy.std also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            axis or axes along which standard deviation must be computed.
            By default, the product must be computed over the entire array.
            If a tuple of integers, products must be computed over multiple
            axes. Default: ``None``.
        correction
            degrees of freedom adjustment. Setting this parameter to a
            value other than ``0`` has the effect of adjusting the
            divisor during the calculation of the standard deviation
            according to ``N-c`` where ``N`` corresponds to the total
            number of elements over which the standard deviation is
            computed and ``c`` corresponds to the provided degrees of
            freedom adjustment. When computing the standard deviation
            of a population, setting this parameter to ``0`` is the
            standard choice (i.e., the provided array contains data
            constituting an entire population). When computing
            the corrected sample standard deviation, setting this
            parameter to ``1`` is the standard choice (i.e., the
            provided array contains data sampled from a larger
            population; this is commonly referred to as Bessel's
            correction). Default: ``0``.

        keepdims
            bool, if True, the reduced axes (dimensions) must be
            included in the result as singleton dimensions, and,
            accordingly, the result must be compatible with the
            input array (see Broadcasting). Otherwise, if False,
            the reduced axes (dimensions) must not be included in
            the result. Default: ``False``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            container, if the product was computed over the entire array,
            a zero-dimensional array containing the product;
            otherwise, a non-zero-dimensional array containing the products.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        With: class: `ivy.Array` input:

        >>> x = ivy.array([-1., 0., 1.])
        >>> y = x.std()
        >>> print(y)
        ivy.array(0.81649661)

        >>> x = ivy.array([-1., 0., 1.])
        >>> z = x.std(correction=1)
        >>> print(z)
        ivy.array(1.)

        >>> x = ivy.array([[0., 4.]])
        >>> y = x.std(keepdims=True)
        >>> print(y)
        ivy.array([[2.]])

        >>> x = ivy.array([2., 1.])
        >>> y = ivy.array(0.)
        >>> x.std(out=y)
        >>> print(y)
        ivy.array(0.5)

        >>> x = ivy.array([[-1., -2.], [3., 3.]])
        >>> y = x.std(axis=1)
        >>> print(y)
        ivy.array([0.5, 0. ])
        """
        return ivy.std(
            self, axis=axis, correction=correction, keepdims=keepdims, out=out
        )

    # Extra #
    # ----- #

    def cumsum(
        self: ivy.Array,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.cumsum. This method simply
        wraps the function, and so the docstring for ivy.cumsum also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input array to apply cumsum.
        axis
            Axis along which the cumulative sum is computed. Default is ``0``.
        exclusive
            Whether to perform cumsum exclusively. Default is ``False``.
        reverse
            Whether to perform the cumsum from last to first element in the selected
            axis. Default is ``False`` (from first to last element)
        dtype
            Data type of the returned array. Default is ``None``.
        out
            Optional array container. Default is ``None``.

        Returns
        -------
        ret
            Array which holds the result of applying cumsum at each
            original array elements along the specified axis.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3, 4, 5])
        >>> y = x.cumsum()
        >>> print(y)
        ivy.array([ 1,  3,  6, 10, 15])

        >>> x = ivy.array([2, 6, 4, 10])
        >>> y = x.cumsum(axis=0, exclusive=False, reverse=True, dtype='float64')
        >>> print(y)
        ivy.array([22., 20., 14., 10.])

        >>> x = ivy.array([[2, 3], [4, 6], [8, 12]])
        >>> y = ivy.zeros((3, 2))
        >>> x.cumsum(axis=1, exclusive=True, reverse=False, out=y)
        >>> print(y)
        ivy.array([[0, 2],
                   [0, 4],
                   [0, 8]])

        >>> x = ivy.array([[1, 5, 2],
        ...                [4, 3, 0],
        ...                [4, 8, 2]])
        >>> y = x.cumsum(axis=1, exclusive=True, reverse=True)
        >>> print(y)
        ivy.array([[ 7,  2,  0],
                   [ 3,  0,  0],
                   [10,  2,  0]])

        >>> x = ivy.array([[1, 5, 10], [4, 8, 10], [2, 3, 5]])
        >>> x.cumsum(axis=0, out=x)
        >>> print(x)
        ivy.array([[ 1,  5, 10],
                   [ 5, 13, 20],
                   [ 7, 16, 25]])
        """
        return ivy.cumsum(self._data, axis, exclusive, reverse, dtype=dtype, out=out)

    def cumprod(
        self: ivy.Array,
        /,
        *,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.cumprod. This method simply
        wraps the function, and so the docstring for ivy.cumprod also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array
        axis
            int, axis along which to take the cumulative product. Default is ``0``.
        exclusive
            optional bool, whether to exclude the first value of the input array.
            Default is ``False``.
        reverse
            Whether to perform the cumprod from last to first element in the selected
            axis. Default is ``False`` (from first to last element)
        dtype
            data type of the returned array. If None, if the default data type
            corresponding to the data type “kind” (integer or floating-point) of x
            has a smaller range of values than the data type of x (e.g., x has data
            type int64 and the default data type is int32, or x has data type uint64
            and the default data type is int64), the returned array must have the
            same data type as x. if x has a floating-point data type, the returned array
            must have the default floating-point data type. if x has a signed integer
            data type (e.g., int16), the returned array must have the default integer
            data type. if x has an unsigned integer data type (e.g., uint16), the
            returned array must have an unsigned integer data type having the same
            number of bits as the default integer data type (e.g., if the default
            integer data type is int32, the returned array must have a uint32 data
            type). If the data type (either specified or resolved) differs from the
            data type of x, the input array should be cast to the specified data type
            before computing the product. Default: ``None``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Input array with cumulatively multiplied elements along the specified axis.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3, 4, 5])
        >>> y = x.cumprod()
        >>> print(y)
        ivy.array([1, 2, 6, 24, 120])

        >>> x = ivy.array([[2, 3], [5, 7], [11, 13]])
        >>> y = ivy.zeros((3, 2), dtype="int32")
        >>> x.cumprod(axis=1, exclusive=True, out=y)
        >>> print(y)
        ivy.array([[0, 0],
                   [0, 0],
                   [0, 0]])
        """
        return ivy.cumprod(
            self._data,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            dtype=dtype,
            out=out,
        )

    def einsum(
        self: ivy.Array,
        equation: str,
        *operands: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.einsum. This method simply
        wraps the function, and so the docstring for ivy.einsum also applies to
        this method with minimal changes.

        Parameters
        ----------
        equation
            A str describing the contraction, in the same format as numpy.einsum.
        operands
            seq of arrays, the inputs to contract (each one an ivy.Array), whose shapes
            should be consistent with equation.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The array with sums computed.

        Examples
        --------
        >>> x = ivy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        >>> y = x.einsum('ii')
        >>> print(y)
        ivy.array(12)

        >>> x = ivy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        >>> z = x.einsum('ij -> j')
        >>> print(z)
        ivy.array([ 9, 12, 15])

        >>> A = ivy.array([0, 1, 2])
        >>> B = ivy.array([[ 0,  1,  2,  3],
        ...                [ 4,  5,  6,  7],
        ...                [ 8,  9, 10, 11]])
        >>> C = A.einsum('i,ij->i', B)
        >>> print(C)
        ivy.array([ 0, 22, 76])

        >>> A = ivy.array([[1, 1, 1],
        ...                [2, 2, 2],
        ...                [5, 5, 5]])
        >>> B = ivy.array([[0, 1, 0],
        ...                [1, 1, 0],
        ...                [1, 1, 1]])
        >>> C = A.einsum('ij,jk->ik', B)
        >>> print(C)
        ivy.array([[ 2,  3,  1],
               [ 4,  6,  2],
               [10, 15,  5]])

        >>> A = ivy.arange(10)
        >>> B = A.einsum('i->')
        >>> print(B)
        ivy.array(45)

        >>> A = ivy.arange(10)
        >>> B = ivy.arange(5, 15)
        >>> C = A.einsum('i,i->i', B)
        >>> print(C)
        ivy.array([  0,   6,  14,  24,  36,  50,  66,  84, 104, 126])

        >>> A = ivy.arange(10)
        >>> B = ivy.arange(5, 15)
        >>> C = A.einsum('i,i->', B) # or just use 'i,i'
        >>> print(C)
        ivy.array(510)
        """
        return ivy.einsum(equation, *(self._data,) + operands, out=out)
