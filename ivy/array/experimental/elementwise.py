# global
import abc
from typing import Optional, Union, Tuple, List

# local
import ivy


class ArrayWithElementWiseExperimental(abc.ABC):
    def sinc(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sinc. This method simply wraps the
        function, and so the docstring for ivy.sinc also applies to this method
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

    def lcm(
        self: ivy.Array, x2: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array.
        x2
            second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            an array that includes the element-wise least common multiples
            of 'self' and x2

        Examples
        --------
        >>> x1=ivy.array([2, 3, 4])
        >>> x2=ivy.array([5, 8, 15])
        >>> x1.lcm(x2)
        ivy.array([10, 21, 60])
        """
        return ivy.lcm(self, x2, out=out)

    def fmod(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.fmod. This method simply
        wraps the function, and so the docstring for ivy.fmod also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
        x1
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
        """ivy.Array instance method variant of ivy.fmax. This method simply
        wraps the function, and so the docstring for ivy.fmax also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
        x1
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

    def trapz(
        self: ivy.Array,
        /,
        *,
        x: Optional[ivy.Array] = None,
        dx: Optional[float] = 1.0,
        axis: Optional[int] = -1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.trapz. This method simply
        wraps the function, and so the docstring for ivy.trapz also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The array that should be integrated.
        x
            The sample points corresponding to the input array values.
            If x is None, the sample points are assumed to be evenly spaced
            dx apart. The default is None.
        dx
            The spacing between sample points when x is None. The default is 1.
        axis
            The axis along which to integrate.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Definite integral of n-dimensional array as approximated along
            a single axis by the trapezoidal rule. If the input array is a
            1-dimensional array, then the result is a float. If n is greater
            than 1, then the result is an n-1 dimensional array.

        Examples
        --------
        >>> y = ivy.array([1, 2, 3])
        >>> ivy.trapz(y)
        4.0
        >>> y = ivy.array([1, 2, 3])
        >>> x = ivy.array([4, 6, 8])
        >>> ivy.trapz(y, x=x)
        8.0
        >>> y = ivy.array([1, 2, 3])
        >>> ivy.trapz(y, dx=2)
        8.0
        """
        return ivy.trapz(self._data, x=x, dx=dx, axis=axis, out=out)

    def float_power(
        self: Union[ivy.Array, float, list, tuple],
        x2: Union[ivy.Array, float, list, tuple],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.float_power. This method simply
        wraps the function, and so the docstring for ivy.float_power also applies to
        this method with minimal changes.

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

    def exp2(
        self: Union[ivy.Array, float, list, tuple],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.exp2. This method simply
        wraps the function, and so the docstring for ivy.exp2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Array-like input.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Element-wise 2 to the power x. This is a scalar if x is a scalar.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> x.exp2()
        ivy.array([2.,    4.,   8.])
        >>> x = [5, 6, 7]
        >>> x.exp2()
        ivy.array([32.,   64.,  128.])
        """
        return ivy.exp2(self._data, out=out)

    def count_nonzero(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.count_nonzero. This method simply
        wraps the function, and so the docstring for ivy.count_nonzero also applies to
        this method with minimal changes.

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
        keepdims: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.nansum. This method simply
        wraps the function, and so the docstring for ivy.nansum also applies to
        this method with minimal changes.

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

    def gcd(
        self: Union[ivy.Array, int, list, tuple],
        x2: Union[ivy.Array, int, list, tuple],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.gcd. This method simply
        wraps the function, and so the docstring for ivy.gcd also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            First array-like input.
        x2
            Second array-like input
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Element-wise gcd of |x1| and |x2|.

        Examples
        --------
        >>> x1 = ivy.array([1, 2, 3])
        >>> x2 = ivy.array([4, 5, 6])
        >>> x1.gcd(x2)
        ivy.array([1.,    1.,   3.])
        >>> x1 = ivy.array([1, 2, 3])
        >>> x1.gcd(10)
        ivy.array([1.,   2.,  1.])
        """
        return ivy.gcd(self._data, x2, out=out)

    def isclose(
        self: ivy.Array,
        b: ivy.Array,
        /,
        *,
        rtol: Optional[float] = 1e-05,
        atol: Optional[float] = 1e-08,
        equal_nan: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.isclose. This method simply
        wraps the function, and so the docstring for ivy.isclose also applies to
        this method with minimal changes.

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

    def isposinf(
        self: Union[ivy.Array, float, list, tuple],
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.isposinf. This method simply
        wraps the function, and so the docstring for ivy.isposinf also applies to
        this method with minimal changes.

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
            Returns a boolean array with values True where
            the corresponding element of the input is positive
            infinity and values False where the element of the
            input is not positive infinity.

        Examples
        --------
        >>> a = ivy.array([12.1, -ivy.inf, ivy.inf])
        >>> ivy.isposinf(a)
        ivy.array([False, False,  True])
        """
        return ivy.isposinf(self._data, out=out)

    def isneginf(
        self: Union[ivy.Array, float, list, tuple],
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.isneginf. This method simply
        wraps the function, and so the docstring for ivy.isneginf also applies to
        this method with minimal changes.

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
            Returns a boolean array with values True where
            the corresponding element of the input is negative
            infinity and values False where the element of the
            input is not negative infinity.

        Examples
        --------
        >>> x = ivy.array([12.1, -ivy.inf, ivy.inf])
        >>> x.isneginf()
        ivy.array([False, True,  False])
        """
        return ivy.isneginf(self._data, out=out)

    def nan_to_num(
        self: ivy.Array,
        /,
        *,
        copy: Optional[bool] = True,
        nan: Optional[Union[float, int]] = 0.0,
        posinf: Optional[Union[float, int]] = None,
        neginf: Optional[Union[float, int]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.nan_to_num. This method simply
        wraps the function, and so the docstring for ivy.nan_to_num also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Array input.
        copy
            Whether to create a copy of x (True) or to replace values in-place (False).
            The in-place operation only occurs if casting to an array does not require
            a copy. Default is True.
        nan
            Value to be used to fill NaN values. If no value is passed then NaN values
            will be replaced with 0.0.
        posinf
            Value to be used to fill positive infinity values. If no value is passed
            then positive infinity values will be replaced with a very large number.
        neginf
            Value to be used to fill negative infinity values.
            If no value is passed then negative infinity values
            will be replaced with a very small (or negative) number.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with the non-finite values replaced.
            If copy is False, this may be x itself.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3, nan])
        >>> x.nan_to_num()
        ivy.array([1.,    1.,   3.,   0.0])
        >>> x = ivy.array([1, 2, 3, inf])
        >>> x.nan_to_num(posinf=5e+100)
        ivy.array([1.,   2.,   3.,   5e+100])
        """
        return ivy.nan_to_num(
            self._data, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=out
        )

    def logaddexp2(
        self: Union[ivy.Array, float, list, tuple],
        x2: Union[ivy.Array, float, list, tuple],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.logaddexp2. This method
        simply wraps the function, and so the docstring for ivy.logaddexp2 also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            First array-like input.
        x2
            Second array-like input
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Element-wise logaddexp2 of x1 and x2.

        Examples
        --------
        >>> x1 = ivy.array([1, 2, 3])
        >>> x2 = ivy.array([4, 5, 6])
        >>> x1.logaddexp2(x2)
        ivy.array([4.169925, 5.169925, 6.169925])
        """
        return ivy.logaddexp2(self._data, x2, out=out)

    def signbit(
        self: Union[ivy.Array, float, int, list, tuple],
        x2: Union[ivy.Array, float, int, list, tuple],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.signbit. This method
        simply wraps the function, and so the docstring for ivy.signbit also
        applies to this method with minimal changes.

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

    def allclose(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        rtol: Optional[float] = 1e-05,
        atol: Optional[float] = 1e-08,
        equal_nan: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
    ) -> bool:
        """
        ivy.Array instance method variant of ivy.allclose. This method simply
        wraps the function, and so the docstring for ivy.allclose also applies to
        this method with minimal changes.

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
        False

        >>> x1 = ivy.array([1.0, ivy.nan])
        >>> x2 = ivy.array([1.0, ivy.nan])
        >>> y = x1.allclose(x2, equal_nan=True)
        >>> print(y)
        True

        >>> x1 = ivy.array([1e-10, 1e-10])
        >>> x2 = ivy.array([1.00001e-10, 1e-10])
        >>> y = x1.allclose(x2, rtol=0.005, atol=0.0)
        >>> print(y)

        """
        return ivy.allclose(
            self._data, x2, rtol=rtol, atol=atol, equal_nan=equal_nan, out=out
        )

    def diff(
        self: Union[ivy.Array, int, float, list, tuple],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.diff. This method simply
        wraps the function, and so the docstring for ivy.diff also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            array-like input.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Returns the n-th discrete difference along the given axis.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 4, 7, 0]),\
                               b=ivy.array([1, 2, 4, 7, 0]))
        >>> ivy.Container.static_diff(x)
        {
            a: ivy.array([ 1,  2,  3, -7])
            b: ivy.array([ 1,  2,  3, -7])
        }
        """
        return ivy.diff(self._data, out=out)

    def fix(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.fix. This method
        simply wraps the function, and so the docstring for ivy.fix also
        applies to this method with minimal changes.

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
        out: Optional[ivy.Container] = None,
    ) -> bool:
        """
        ivy.Array instance method variant of ivy.nextafter. This method simply
        wraps the function, and so the docstring for ivy.nextafter also applies to
        this method with minimal changes.

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
        out: Optional[ivy.Container] = None,
    ) -> bool:
        """
        ivy.Array instance method variant of ivy.zeta. This method simply
        wraps the function, and so the docstring for ivy.zeta also applies to
        this method with minimal changes.

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
        spacing: Optional[Union[int, list, tuple]] = 1,
        edge_order: Optional[int] = 1,
        axis: Optional[Union[int, list, tuple]] = None,
    ) -> Union[ivy.Array, List[ivy.Array]]:
        """Calculates gradient of x with respect to (w.r.t.) spacing

        Parameters
        ----------
        x
            input array representing outcomes of the function
            spacing
            if not given, indices of x will be used
            if scalar indices of x will be scaled with this value
            if array gradient of x w.r.t. spacing
        edge_order
            1 or 2, for 'frist order' and 'second order' estimation
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
