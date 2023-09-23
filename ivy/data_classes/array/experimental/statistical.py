# global
import abc
from typing import Optional, Union, Tuple, Sequence

# local
import ivy


class _ArrayWithStatisticalExperimental(abc.ABC):
    def histogram(
        self: ivy.Array,
        /,
        *,
        bins: Optional[Union[int, ivy.Array, ivy.NativeArray, str]] = None,
        axis: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        extend_lower_interval: Optional[bool] = False,
        extend_upper_interval: Optional[bool] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        range: Optional[Tuple[float]] = None,
        weights: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        density: Optional[bool] = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.histogram. This method simply wraps the
        function, and so the docstring for ivy.histogram also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.
        bins
            if ``bins`` is an int, it defines the number of equal-width bins in the
            given range.
            if ``bins`` is an array, it defines a monotonically increasing array of bin
            edges, including the rightmost edge, allowing for non-uniform bin widths.
        axis
            dimension along which maximum values must be computed. By default, the
            maximum value must be computed over the entire array. Default: ``None``.
        extend_lower_interval
            if True, extend the lowest interval I0 to (-inf, c1].
        extend_upper_interval
            ff True, extend the upper interval I_{K-1} to [c_{K-1}, +inf).
        dtype
            the output type.
        range
            the lower and upper range of the bins. The first element of the range must
            be less than or equal to the second.
        weights
            each value in ``a`` only contributes its associated weight towards the bin
            count (instead of 1). Must be of the same shape as a.
        density
            if True, the result is the value of the probability density function at the
            bin, normalized such that the integral over the range of bins is 1.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a tuple containing the values of the histogram and the bin edges.

        Both the description and the type hints above assumes an array input for
        simplicity, but this function is *nestable*, and therefore also accepts
        :class:`ivy.Container` instances in place of any of the arguments.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.array([0, 1, 2])
        >>> y = ivy.array([0., 0.5, 1., 1.5, 2.])
        >>> z = ivy.histogram(x, bins=y)
        >>> print(z)
        ivy.array([1., 0., 1., 1.])
        """
        return ivy.histogram(
            self._data,
            bins=bins,
            axis=axis,
            extend_lower_interval=extend_lower_interval,
            extend_upper_interval=extend_upper_interval,
            dtype=dtype,
            range=range,
            weights=weights,
            density=density,
            out=out,
        )

    def median(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.median. This method simply wraps the
        function, and so the docstring for ivy.median also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        axis
            Axis or axes along which the medians are computed. The default is to compute
            the median along a flattened version of the array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The median of the array elements.

        Examples
        --------
        >>> a = ivy.array([[10, 7, 4], [3, 2, 1]])
        >>> a.median()
        3.5
        >>> a.median(axis=0)
        ivy.array([6.5, 4.5, 2.5])
        """
        return ivy.median(self._data, axis=axis, keepdims=keepdims, out=out)

    def nanmean(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.nanmean. This method simply wraps the
        function, and so the docstring for ivy.nanmean also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        axis
            Axis or axes along which the means are computed.
            The default is to compute the mean of the flattened array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one. With this option, the result will broadcast
            correctly against the original a. If the value is anything but the default,
            then keepdims will be passed through to the mean or sum methods of
            sub-classes of ndarray. If the sub-classes methods does not implement
            keepdims any exceptions will be raised.
        dtype
            The desired data type of returned tensor. Default is None.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The nanmean of the array elements.

        Examples
        --------
        >>> a = ivy.array([[1, ivy.nan], [3, 4]])
        >>> a.nanmean()
        2.6666666666666665
        >>> a.nanmean(axis=0)
        ivy.array([2.,  4.])
        """
        return ivy.nanmean(
            self._data, axis=axis, keepdims=keepdims, dtype=dtype, out=out
        )

    def nanprod(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[Union[int, float, complex]] = None,
        where: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.nanprod. This method simply wraps the
        function, and so the docstring for ivy.prod also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        axis
            Axis or axes along which the product is computed.
            The default is to compute the product of the flattened array.
        dtype
            The desired data type of returned array. Default is None.
        out
            optional output array, for writing the result to.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one. With this option, the result will broadcast
            correctly against the original a.
        initial
            The starting value for this product.
        where
            Elements to include in the product

        Returns
        -------
        ret
            The product of array elements over a given axis treating
            Not a Numbers (NaNs) as ones

        Examples
        --------
        >>> a = ivy.array([[1, 2], [3, ivy.nan]])
        >>> a.nanprod(a)
        6.0
        >>> a.nanprod(a, axis=0)
        ivy.array([3., 2.])
        """
        return ivy.nanprod(
            self._data,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            out=out,
            initial=initial,
            where=where,
        )

    def quantile(
        self: ivy.Array,
        q: Union[ivy.Array, float],
        /,
        *,
        axis: Optional[Union[Sequence[int], int]] = None,
        keepdims: bool = False,
        interpolation: str = "linear",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.quantile. This method simply wraps the
        function, and so the docstring for ivy.quantile also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        q
            Quantile or sequence of quantiles to compute, which must be
            between 0 and 1 inclusive.
        axis
            Axis or axes along which the quantiles are computed. The default
            is to compute the quantile(s) along a flattened version of the array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one. With this option, the result will broadcast
            correctly against the original array a.
        interpolation
            {'nearest', 'linear', 'lower', 'higher', 'midpoint'}. Default value:
            'linear'.
            This specifies the interpolation method to use when the desired quantile
            lies between two data points i < j:
            - linear: i + (j - i) * fraction, where fraction is the fractional part of
            the index surrounded by i and j.
            - lower: i.
            - higher: j.
            - nearest: i or j, whichever is nearest.
            - midpoint: (i + j) / 2. linear and midpoint interpolation do not work with
            integer dtypes.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            A (rank(q) + N - len(axis)) dimensional array of same dtype as a, or,
            if axis is None, a rank(q) array. The first rank(q) dimensions index
            quantiles for different values of q.

        Examples
        --------
        >>> a = ivy.array([[10., 7., 4.], [3., 2., 1.]])
        >>> q = ivy.array(0.5)
        >>> a.quantile(q)
        ivy.array(3.5)

        >>> a = ivy.array([[10., 7., 4.], [3., 2., 1.]])
        >>> q = 0.5
        >>> a.quantile(q)
        ivy.array(3.5)

        >>> a.quantile(q, axis=0)
        ivy.array([6.5, 4.5, 2.5])

        >>> a.quantile(q, axis=1)
        ivy.array([7.,  2.])

        >>> a.quantile(q, axis=1, keepdims=True)
        ivy.array([[7.],[2.]])

        >>> a = ivy.array([1., 2., 3., 4.])
        >>> q = ivy.array([0.3, 0.7])
        >>> a.quantile(q, interpolation='lower')
        ivy.array([1., 3.])
        """
        return ivy.quantile(
            self._data,
            q,
            axis=axis,
            keepdims=keepdims,
            interpolation=interpolation,
            out=out,
        )

    def corrcoef(
        self: ivy.Array,
        /,
        *,
        y: Optional[ivy.Array] = None,
        rowvar: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.corrcoef. This method simply wraps the
        function, and so the docstring for ivy.corrcoef also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        y
            An additional input array.
            `y` has the same shape as `x`.
        rowvar
            If rowvar is True (default), then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is transposed:
            each column represents a variable, while the rows contain observations.

        Returns
        -------
        ret
            The corrcoef of the array elements.

        Examples
        --------
        >>> a = ivy.array([[0., 1., 2.], [2., 1., 0.]])
        >>> a.corrcoef()
            ivy.array([[ 1., -1.],
                       [-1.,  1.]])
        >>> a.corrcoef(rowvar=False)
            ivy.array([[ 1., nan, -1.],
                       [nan, nan, nan],
                       [-1., nan,  1.]])
        """
        return ivy.corrcoef(self._data, y=y, rowvar=rowvar, out=out)

    def nanmedian(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        overwrite_input: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.nanmedian. This method simply wraps the
        function, and so the docstring for ivy.nanmedian also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input array.
        axis
            The axis or axes along which the means are computed.
            The default is to compute the mean of the flattened array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one. With this option, the result will broadcast
            correctly against the original input array. If the value is anything
            but the default, then keepdims will be passed through to the mean or
            sum methods of sub-classes of ndarray. If the sub-classes methods does
            not implement keepdims any exceptions will be raised.
        overwrite_input
            If True, then allow use of memory of input array a for calculations.
            The input array will be modified by the call to median. This will
            save memory when you do not need to
            preserve the contents of the input array.
            Treat the input as undefined, but it will
            probably be fully or partially sorted.
            Default is False. If overwrite_input
            is True and input array is not already an ndarray,
            an error will be raised.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            A new array holding the result. If the input contains integers

        Examples
        --------
        With :class:`ivy.array` input and default backend set as `numpy`:

        >>> a = ivy.array([[10.0, ivy.nan, 4], [3, 2, 1]])
        >>> a.nanmedian()
            ivy.array(3.)
        >>> a.nanmedian(axis=0)
            ivy.array([6.5, 2. , 2.5])
        """
        return ivy.nanmedian(
            self._data,
            axis=axis,
            keepdims=keepdims,
            overwrite_input=overwrite_input,
            out=out,
        )

    def bincount(
        self,
        /,
        *,
        weights: Optional[ivy.Array] = None,
        minlength: int = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.bincount. This method simply wraps the
        function, and so the docstring for ivy.bincount also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array. The array is flattened if it is not already 1-dimensional.
        weights
            Optional weights, array of the same shape as self.
        minlength
            A minimum number of bins for the output array.
        out
            An array of the same shape as the returned array, or of the shape
            (minlength,) if minlength is specified.

        Returns
        -------
        ret
            The result of binning the input array.

        Examples
        --------
        >>> a = ivy.array([0, 1, 1, 3, 2, 1, 7])
        >>> a.bincount()
            ivy.array([1, 3, 1, 1, 0, 0, 0, 1])
        >>> a.bincount(minlength=10)
            ivy.array([1, 3, 1, 1, 0, 0, 0, 1, 0, 0])
        >>> a.bincount(weights=ivy.array([0.3, 0.5, 0.2, 0.7, 1., 0.6, 1.]))
            ivy.array([0.3, 1.3, 1. , 0.7, 0. , 0. , 0. , 1. ])
        """
        return ivy.bincount(
            self._data,
            weights=weights,
            minlength=minlength,
            out=out,
        )

    def igamma(
        self: ivy.Array,
        /,
        *,
        x: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.igamma. This method simply wraps the
        function, and so the docstring for ivy.igamma also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        x
            An additional input array.
            `x` has the same type as `a`.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The lower incomplete gamma function of the array elements.

        Examples
        --------
        >>> a = ivy.array([2.5])
        >>> x = ivy.array([1.7, 1.2])
        >>> a.igamma(x)
            ivy.array([0.3614, 0.2085])
        """
        return ivy.igamma(
            self._data,
            x=x,
            out=out,
        )

    def cov(
        self: ivy.Array,
        x2: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        /,
        *,
        rowVar: bool = True,
        bias: bool = False,
        ddof: Optional[int] = None,
        fweights: Optional[ivy.Array] = None,
        aweights: Optional[ivy.Array] = None,
        dtype: Optional[type] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cov. This method simply wraps the
        function, and so the docstring for ivy.cov also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            a 1D or 2D input array, with a numeric data type.
        x2
            optional second 1D or 2D input array, with a numeric data type.
            Must have the same shape as ``self``.
        rowVar
            optional variable where each row of input is interpreted as a variable
            (default = True). If set to False, each column is instead interpreted as a
            variable.
        bias
            optional variable for normalizing input (default = False) by (N - 1) where
            N is the number of given observations. If set to True, then normalization
            is instead by N. Can be overridden by keyword ``ddof``.
        ddof
            optional variable to override ``bias`` (default = None). ddof=1 will return
            the unbiased estimate, even with fweights and aweights given. ddof=0 will
            return the simple average.
        fweights
            optional 1D array of integer frequency weights; the number of times each
            observation vector should be repeated.
        aweights
            optional 1D array of observation vector weights. These relative weights are
            typically large for observations considered "important" and smaller for
            observations considered less "important". If ddof=0 is specified, the array
            of weights can be used to assign probabilities to observation vectors.
        dtype
            optional variable to set data-type of the result. By default, data-type
            will have at least ``float64`` precision.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the covariance matrix of an input matrix, or the
            covariance matrix of two variables. The returned array must have a
            floating-point data type determined by Type Promotion Rules and must be
            a square matrix of shape (N, N), where N is the number of variables in the
            input(s).

        Examples
        --------
        >>> x = ivy.array([[1, 2, 3],
        ...                [4, 5, 6]])
        >>> y = x[0].cov(x[1])
        >>> print(y)
        ivy.array([[1., 1.],
               [1., 1.]])

        >>> x = ivy.array([1,2,3])
        >>> y = ivy.array([4,5,6])
        >>> z = x.cov(y)
        >>> print(z)
        ivy.array([[1., 1.],
               [1., 1.]])
        """
        return ivy.cov(
            self._data,
            x2,
            rowVar=rowVar,
            bias=bias,
            ddof=ddof,
            fweights=fweights,
            aweights=aweights,
            dtype=dtype,
        )

    def cummax(
        self: ivy.Array,
        /,
        *,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cummax. This method simply wraps the
        function, and so the docstring for ivy.cummax also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array
        axis
            int, axis along which to take the cumulative maximum. Default is ``0``.
        reverse
            Whether to perform the cummax from last to first element in the selected
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
        --------
        >>> x = ivy.array([1, 2, 5, 4, 3])
        >>> y = x.cummax()
        >>> print(y)
        (ivy.array([1, 2, 5, 5, 5]), ivy.array([0, 1, 2, 2, 2]))

        >>> x = ivy.array([[2, 3], [5, 7], [11, 13]])
        >>> y = ivy.zeros((3, 2), dtype="int32")
        >>> x.cummax(axis=1, reverse=True, out=y)
        >>> print(y)
        ivy.array([[0, 0],
               [0, 0],
               [0, 0]])
        """
        return ivy.cummax(
            self._data,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            dtype=dtype,
            out=out,
        )

    def cummin(
        self: ivy.Array,
        /,
        *,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cummin. This method simply wraps the
        function, and so the docstring for ivy.cummin also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array
        axis
            int, axis along which to take the cumulative minimum. Default is ``0``.
        reverse
            Whether to perform the cummin from last to first element in the selected
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
        --------
        >>> x = ivy.array([1, 2, 3, 4, 5])
        >>> y = x.cummin()
        >>> print(y)
        ivy.array([1, 1, 1, 1, 1])

        >>> x = ivy.array([[2, 3], [5, 7], [11, 13]])
        >>> y = ivy.zeros((3, 2), dtype="int32")
        >>> x.cummin(axis=1, reverse=True, out=y)
        >>> print(y)
        ivy.array([[ 2,  3],
                  [ 5,  7],
                  [11, 13]])
        """
        return ivy.cummin(
            self._data,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            dtype=dtype,
            out=out,
        )
