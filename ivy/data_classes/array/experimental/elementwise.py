# global
import abc
from typing import Optional, Union, Tuple, List
from numbers import Number

# local
import ivy


class _ArrayWithElementWiseExperimental(abc.ABC):
    def lgamma(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.lgamma. This method simply wraps the
        function, and so the docstring for ivy.lgamma also applies to this method with
        minimal changes.

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
        >>> x = ivy.array([1., 2., 3.])
        >>> y = x.lgamma()
        >>> print(y)
        ivy.array([0., 0., 0.69314718])

        >>> x = ivy.array([4.5, -4, -5.6])
        >>> x.lgamma(out = x)
        >>> print(x)
        ivy.array([2.45373654, inf, -4.6477685 ])
        """
        return ivy.lgamma(self._data, out=out)

    def sinc(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sinc. This method simply wraps the
        function, and so the docstring for ivy.sinc also applies to this method with
        minimal changes.

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
            an array containing the sinc of each element in ``self``. The returned
            array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0.5, 1.5, 2.5, 3.5])
        >>> y = x.sinc()
        >>> print(y)
        ivy.array([0.637,-0.212,0.127,-0.0909])
        """
        return ivy.sinc(self._data, out=out)

    def fmod(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.fmod. This method simply wraps the
        function, and so the docstring for ivy.fmod also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            First input array.
        x2
            Second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = ivy.array([2, 3, 4])
        >>> x2 = ivy.array([1, 5, 2])
        >>> x1.fmod(x2)
        ivy.array([ 0,  3,  0])

        >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
        >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
        >>> x1.fmod(x2)
        ivy.array([ nan,  nan,  nan])
        """
        return ivy.fmod(self._data, x2, out=out)

    def fmax(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.fmax. This method simply wraps the
        function, and so the docstring for ivy.fmax also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            First input array.
        x2
            Second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with element-wise maximums.

        Examples
        --------
        >>> x1 = ivy.array([2, 3, 4])
        >>> x2 = ivy.array([1, 5, 2])
        >>> ivy.fmax(x1, x2)
        ivy.array([ 2.,  5.,  4.])

        >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
        >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
        >>> x1.fmax(x2)
        ivy.array([ 0,  0,  nan])
        """
        return ivy.fmax(self._data, x2, out=out)

    def float_power(
        self: Union[ivy.Array, float, list, tuple],
        x2: Union[ivy.Array, float, list, tuple],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.float_power. This method simply wraps
        the function, and so the docstring for ivy.float_power also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Array-like with elements to raise in power.
        x2
            Array-like of exponents. If x1.shape != x2.shape,
            they must be broadcastable to a common shape
            (which becomes the shape of the output).
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The bases in x1 raised to the exponents in x2.
            This is a scalar if both x1 and x2 are scalars

        Examples
        --------
        >>> x1 = ivy.array([1, 2, 3, 4, 5])
        >>> x1.float_power(3)
        ivy.array([1.,    8.,   27.,   64.,  125.])
        >>> x1 = ivy.array([1, 2, 3, 4, 5])
        >>> x2 = ivy.array([2, 3, 3, 2, 1])
        >>> x1.float_power(x2)
        ivy.array([1.,   8.,  27.,  16.,   5.])
        """
        return ivy.float_power(self._data, x2, out=out)

    def copysign(
        self: Union[ivy.Array, ivy.NativeArray, Number],
        x2: Union[ivy.Array, ivy.NativeArray, Number],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.copysign. This method simply wraps the
        function, and so the docstring for ivy.copysign also applies to this method with
        minimal changes.

        Parameters
        ----------
        x1
            Array or scalar to change the sign of
        x2
            Array or scalar from which the new signs are applied
            Unsigned zeroes are considered positive.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            x1 with the signs of x2.
            This is a scalar if both x1 and x2 are scalars.

        Examples
        --------
        >>> x1 = ivy.array([0, 1, 2, 3])
        >>> x2 = ivy.array([-1, 1, -2, 2])
        >>> x1.copysign(x2)
        ivy.array([-0.,  1., -2.,  3.])
        >>> x2.copysign(-1)
        ivy.array([-1., -1., -2., -2.])
        """
        return ivy.copysign(self._data, x2, out=out)

    def count_nonzero(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.count_nonzero. This method simply wraps
        the function, and so the docstring for ivy.count_nonzero also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array for which to count non-zeros.
        axis
            optional axis or tuple of axes along which to count non-zeros. Default is
            None, meaning that non-zeros will be counted along a flattened
            version of the input array.
        keepdims
            optional, if this is set to True, the axes that are counted are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.
        dtype
            optional output dtype. Default is of type integer.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
           Number of non-zero values in the array along a given axis. Otherwise,
           the total number of non-zero values in the array is returned.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> x.count_nonzero()
        ivy.array(3)
        >>> x = ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> x.count_nonzero(axis=0)
        ivy.array([[1, 2],
               [2, 2]])
        >>> x = ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> x.count_nonzero(axis=(0,1), keepdims=True)
        ivy.array([[[3, 4]]])
        """
        return ivy.count_nonzero(
            self._data, axis=axis, keepdims=keepdims, dtype=dtype, out=out
        )

    def nansum(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[tuple, int]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.nansum. This method simply wraps the
        function, and so the docstring for ivy.nansum also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        axis
            Axis or axes along which the sum is computed.
            The default is to compute the sum of the flattened array.
        dtype
            The type of the returned array and of the accumulator in
            which the elements are summed. By default, the dtype of input is used.
        keepdims
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        >>> a = ivy.array([[ 2.1,  3.4,  ivy.nan], [ivy.nan, 2.4, 2.1]])
        >>> ivy.nansum(a)
        10.0
        >>> ivy.nansum(a, axis=0)
        ivy.array([2.1, 5.8, 2.1])
        >>> ivy.nansum(a, axis=1)
        ivy.array([5.5, 4.5])
        """
        return ivy.nansum(
            self._data, axis=axis, dtype=dtype, keepdims=keepdims, out=out
        )

    def isclose(
        self: ivy.Array,
        b: ivy.Array,
        /,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.isclose. This method simply wraps the
        function, and so the docstring for ivy.isclose also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            First input array.
        b
            Second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in a will be
            considered equal to NaN's in b in the output array.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        >>> a = ivy.array([[ 2.1,  3.4,  ivy.nan], [ivy.nan, 2.4, 2.1]])
        >>> b = ivy.array([[ 2.1,  3.4,  ivy.nan], [ivy.nan, 2.4, 2.1]])
        >>> a.isclose(b)
        ivy.array([[True, True, False],
               [False, True, True]])
        >>> a.isclose(b, equal_nan=True)
        ivy.array([[True, True, True],
               [True, True, True]])
        >>> a=ivy.array([1.0, 2.0])
        >>> b=ivy.array([1.0, 2.001])
        >>> a.isclose(b, atol=0.0)
        ivy.array([True, False])
        >>> a.isclose(b, rtol=0.01, atol=0.0)
        ivy.array([True, True])
        """
        return ivy.isclose(
            self._data, b, rtol=rtol, atol=atol, equal_nan=equal_nan, out=out
        )

    def signbit(
        self: Union[ivy.Array, float, int, list, tuple],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.signbit. This method simply wraps the
        function, and so the docstring for ivy.signbit also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Array-like input.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Element-wise signbit of x.

        Examples
        --------
        >>> x = ivy.array([1, -2, 3])
        >>> x.signbit()
        ivy.array([False, True, False])
        """
        return ivy.signbit(self._data, out=out)

    def hypot(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.hypot. This method simply wraps the
        function, and so the docstring for ivy.hypot also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            First input array
        x2
            Second input array
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array containing the hypotenuse computed from each element of the
            input arrays.

        Examples
        --------
        >>> x = ivy.array([3.0, 4.0, 5.0])
        >>> y = ivy.array([4.0, 5.0, 6.0])
        >>> x.hypot(y)
        ivy.array([5.0, 6.4031, 7.8102])
        """
        return ivy.hypot(self._data, x2, out=out)

    def allclose(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> bool:
        """
        ivy.Array instance method variant of ivy.allclose. This method simply wraps the
        function, and so the docstring for ivy.allclose also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            First input array.
        x2
            Second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in a will be
            considered equal to NaN's in b in the output array.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            Returns True if the two arrays are equal within the given tolerance;
            False otherwise.

        Examples
        --------
        >>> x1 = ivy.array([1e10, 1e-7])
        >>> x2 = ivy.array([1.00001e10, 1e-8])
        >>> y = x1.allclose(x2)
        >>> print(y)
        ivy.array(False)

        >>> x1 = ivy.array([1.0, ivy.nan])
        >>> x2 = ivy.array([1.0, ivy.nan])
        >>> y = x1.allclose(x2, equal_nan=True)
        >>> print(y)
        ivy.array(True)

        >>> x1 = ivy.array([1e-10, 1e-10])
        >>> x2 = ivy.array([1.00001e-10, 1e-10])
        >>> y = x1.allclose(x2, rtol=0.005, atol=0.0)
        >>> print(y)
        ivy.array(True)
        """
        return ivy.allclose(
            self._data, x2, rtol=rtol, atol=atol, equal_nan=equal_nan, out=out
        )

    def diff(
        self: ivy.Array,
        /,
        *,
        n: int = 1,
        axis: int = -1,
        prepend: Optional[Union[ivy.Array, ivy.NativeArray, int, list, tuple]] = None,
        append: Optional[Union[ivy.Array, ivy.NativeArray, int, list, tuple]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.diff. This method simply wraps the
        function, and so the docstring for ivy.diff also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            array-like input.
        n
            The number of times values are differenced. If zero, the input is returned
            as-is.
        axis
            The axis along which the difference is taken, default is the last axis.
        prepend,append
            Values to prepend/append to x along given axis prior to performing the
            difference. Scalar values are expanded to arrays with length 1 in the
            direction of axis and the shape of the input array in along all other
            axes. Otherwise the dimension and shape must match x except along axis.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Returns the n-th discrete difference along the given axis.

        Examples
        --------
        >>> x = ivy.array([1, 2, 4, 7, 0])
        >>> x.diff()
        ivy.array([ 1,  2,  3, -7])
        """
        return ivy.diff(
            self._data, n=n, axis=axis, prepend=prepend, append=append, out=out
        )

    def fix(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.fix. This method simply wraps the
        function, and so the docstring for ivy.fix also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Array input.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array of floats with elements corresponding to input elements
            rounded to nearest integer towards zero, element-wise.

        Examples
        --------
        >>> x = ivy.array([2.1, 2.9, -2.1])
        >>> x.fix()
        ivy.array([ 2.,  2., -2.])
        """
        return ivy.fix(self._data, out=out)

    def nextafter(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.nextafter. This method simply wraps the
        function, and so the docstring for ivy.nextafter also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            First input array.
        x2
            Second input array.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            The next representable values of x1 in the direction of x2.

        Examples
        --------
        >>> x1 = ivy.array([1.0e-50, 2.0e+50])
        >>> x2 = ivy.array([2.0, 1.0])
        >>> x1.nextafter(x2)
        ivy.array([1.4013e-45., 3.4028e+38])
        """
        return ivy.nextafter(self._data, x2, out=out)

    def zeta(
        self: ivy.Array,
        q: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.zeta. This method simply wraps the
        function, and so the docstring for ivy.zeta also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            First input array.
        q
            Second input array.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            Array with values computed from zeta function from
            input arrays' values.

        Examples
        --------
        >>> x = ivy.array([5.0, 3.0])
        >>> q = ivy.array([2.0])
        >>> x.zeta(q)
        ivy.array([0.0369, 0.2021])
        """
        return ivy.zeta(self._data, q, out=out)

    def gradient(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        spacing: Union[int, list, tuple] = 1,
        edge_order: int = 1,
        axis: Optional[Union[int, list, tuple]] = None,
    ) -> Union[ivy.Array, List[ivy.Array]]:
        """
        Calculate gradient of x with respect to (w.r.t.) spacing.

        Parameters
        ----------
        self
            input array representing outcomes of the function
        spacing
            if not given, indices of x will be used
            if scalar indices of x will be scaled with this value
            if array gradient of x w.r.t. spacing
        edge_order
            1 or 2, for 'first order' and 'second order' estimation
            of boundary values of gradient respectively.
            Note: jax supports edge_order=1 case only
        axis
            dimension(s) to approximate the gradient over
            by default partial gradient is computed in every dimention


        Returns
        -------
        ret
            Array with values computed from gradient function from
            inputs

        Examples
        --------
        >>> spacing = (ivy.array([-2., -1., 1., 4.]),)
        >>> x = ivy.array([4., 1., 1., 16.], )
        >>> ivy.gradient(x, spacing=spacing)
        ivy.array([-3., -2.,  2.,  5.])

        >>> x = ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]])
        >>> ivy.gradient(x)
        [ivy.array([[ 9., 18., 36., 72.],
           [ 9., 18., 36., 72.]]), ivy.array([[ 1. ,  1.5,  3. ,  4. ],
           [10. , 15. , 30. , 40. ]])]

        >>> x = ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]])
        >>> ivy.gradient(x, spacing=2.0)
        [ivy.array([[ 4.5,  9. , 18. , 36. ],
           [ 4.5,  9. , 18. , 36. ]]), ivy.array([[ 0.5 ,  0.75,  1.5 ,  2.  ],
           [ 5.  ,  7.5 , 15.  , 20.  ]])]

        >>> x = ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]])
        >>> ivy.gradient(x, axis=1)
        ivy.array([[ 1. ,  1.5,  3. ,  4. ],
           [10. , 15. , 30. , 40. ]])

        >>> x = ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]])
        >>> ivy.gradient(x, spacing=[3., 2.])
        [ivy.array([[ 3.,  6., 12., 24.],
           [ 3.,  6., 12., 24.]]), ivy.array([[ 0.5 ,  0.75,  1.5 ,  2.  ],
           [ 5.  ,  7.5 , 15.  , 20.  ]])]

        >>> spacing = (ivy.array([0, 2]), ivy.array([0, 3, 6, 9]))
        >>> ivy.gradient(x, spacing=spacing)
        [ivy.array([[ 4.5,  9. , 18. , 36. ],
           [ 4.5,  9. , 18. , 36. ]]), ivy.array([[ 0.33333333,  0.5,  1., 1.33333333],
           [ 3.33333333,  5.        , 10.        , 13.33333333]])]
        """
        return ivy.gradient(
            self._data, spacing=spacing, axis=axis, edge_order=edge_order
        )

    def xlogy(
        self: ivy.Array,
        y: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.xlogy. This method simply wraps the
        function, and so the docstring for ivy.xlogy also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            First input array.
        y
            Second input array.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            The next representable values of x1 in the direction of x2.

        Examples
        --------
        >>> x = ivy.zeros(3)
        >>> y = ivy.array([-1.0, 0.0, 1.0])
        >>> x.xlogy(y)
        ivy.array([0.0, 0.0, 0.0])

        >>> x = ivy.array([1.0, 2.0, 3.0])
        >>> y = ivy.array([3.0, 2.0, 1.0])
        >>> x.xlogy(y)
        ivy.array([1.0986, 1.3863, 0.0000])
        """
        return ivy.xlogy(self._data, y, out=out)

    def binarizer(
        self: ivy.Array, /, *, threshold: float = 0, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        Map the values of the input tensor to either 0 or 1, element-wise, based on the
        outcome of a comparison against a threshold value.

        Parameters
        ----------
        self
             Data to be binarized
        threshold
             Values greater than this are
             mapped to 1, others to 0.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Binarized output data
        """
        return ivy.binarizer(self._data, threshold=threshold, out=out)

    def conj(self: ivy.Array, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.conj. This method simply wraps the
        function, and so the docstring for ivy.conj also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the complex conjugates of values in the input array,
            with the same dtype as the input array.

        Examples
        --------
        >>> x = ivy.array([4+3j, 6+2j, 1-6j])
        >>> x.conj()
        ivy.array([4-3j, 6-2j, 1+6j])
        """
        return ivy.conj(self._data, out=out)

    def lerp(
        self: ivy.Array,
        end: ivy.Array,
        weight: Union[ivy.Array, float],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.lerp. This method simply wraps the
        function, and so the docstring for ivy.lerp also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Array of starting points
        end
            Array of ending points
        weight
            Weight for the interpolation formula  , array or scalar.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            The linear interpolation between array self and array end based on
            scalar or array weight
            self + ((end - self) * weight)

        Examples
        --------
        >>> x = ivy.array([1.0, 2.0, 3.0, 4.0])
        >>> end = ivy.array([10.0, 10.0, 10.0, 10.0])
        >>> weight = 0.5
        >>> x.lerp(end, weight)
        ivy.array([5.5, 6. , 6.5, 7. ])
        """
        return ivy.lerp(self, end, weight, out=out)

    def ldexp(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.ldexp. This method simply wraps the
        function, and so the docstring for ivy.ldexp also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        x2
            The array of exponents.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            The next representable values of x1 in the direction of x2.

        Examples
        --------
        >>> x = ivy.array([1.0, 2.0, 3.0])
        >>> y = ivy.array([3.0, 2.0, 1.0])
        >>> x.ldexp(y)
        ivy.array([8.0, 8.0, 6.0])
        """
        return ivy.ldexp(self._data, x2, out=out)

    def frexp(
        self: ivy.Array, /, *, out: Optional[Tuple[ivy.Array, ivy.Array]] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.frexp. This method simply wraps the
        function, and so the docstring for ivy.frexp also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            The next representable values of x1 in the direction of x2.

        Examples
        --------
        >>> x = ivy.array([1.0, 2.0, 3.0])
        >>> x.frexp()
        ivy.array([[0.5, 0.5, 0.75], [1, 2, 2]])
        """
        return ivy.frexp(self._data, out=out)

    def modf(
        self: ivy.Array, /, *, out: Optional[Tuple[ivy.Array, ivy.Array]] = None
    ) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.modf. This method simply wraps the
        function, and so the docstring for ivy.modf also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        out
            Alternate output arrays in which to place the result.
            The default is None.

        Returns
        -------
        ret
            The fractional and integral parts of the input array.

        Examples
        --------
        >>> x = ivy.array([1.5, 2.7, 3.9])
        >>> x.modf()
        (ivy.array([0.5, 0.7, 0.9]), ivy.array([1, 2, 3]))
        """
        return ivy.modf(self._data, out=out)

    def digamma(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.digamma. This method simply wraps the
        function, and so the docstring for ivy.digamma also applies to this method with
        minimal changes.

        Note
        ----
        The Ivy version only accepts real-valued inputs.

        Parameters
        ----------
        self
            Input array.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            Array with values computed from digamma function from
            input arrays' values, element-wise.

        Examples
        --------
        >>> x = ivy.array([.9, 3, 3.2])
        >>> y = ivy.digamma(x)
        ivy.array([-0.7549271   0.92278427  0.9988394])
        """
        return ivy.digamma(self._data, out=out)

    def sparsify_tensor(
        self: ivy.Array,
        card: int,
    ) -> ivy.Array:
        """
        ivy.Array class method variant of ivy.sparsify_tensor. This method simply wraps
        the function, and so the docstring for ivy.sparsify_tensor also applies to this
        method with minimal changes.

        Parameters
        ----------
        self : array
            The tensor to sparsify.
        card : int
            The number of values to keep.

        Returns
        -------
        ret : array
            The sparsified tensor.

        Examples
        --------
        >>> x = ivy.arange(100)
        >>> x = ivy.reshape(x, (10, 10))
        >>> x.sparsify_tensor(10)
        ivy.array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
        """
        return ivy.sparsify_tensor(self._data, card)
