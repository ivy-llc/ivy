# global
import abc
from typing import Optional, Union, Tuple, Sequence

# local
import ivy


class _ArrayWithStatisticalExperimental(abc.ABC):
    def median(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.median. This method simply
        wraps the function, and so the docstring for ivy.median also applies to
        this method with minimal changes.

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
        """ivy.Array instance method variant of ivy.nanmean. This method simply
        wraps the function, and so the docstring for ivy.nanmean also applies to
        this method with minimal changes.

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

    def unravel_index(
        self: ivy.Array,
        shape: Tuple[int],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.unravel_index. This method simply
        wraps the function, and so the docstring for ivy.unravel_index also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        shape
            The shape of the array to use for unraveling indices.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Tuple with arrays that have the same shape as the indices array.

        Examples
        --------
        >>> indices = ivy.array([22, 41, 37])
        >>> indices.unravel_index((7,6))
        (ivy.array([3, 6, 6]), ivy.array([4, 5, 1]))
        """
        return ivy.unravel_index(self._data, shape, out=out)

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
        """ivy.Array instance method variant of ivy.quantile.
        This method simply wraps the function, and so the docstring
        for ivy.quantile also applies to this method with minimal
        changes.

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
        """ivy.Array instance method variant of ivy.corrcoef. This method simply
        wraps the function, and so the docstring for ivy.corrcoef also applies to
        this method with minimal changes.

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
        """ivy.Array instance method variant of ivy.nanmedian. This method simply
        wraps the function, and so the docstring for ivy.nanmedian also applies to
        this method with minimal changes.

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
        overwrite_input
            If True, then allow use of memory of input array a for calculations.
            The input array will be modified by the call to median. This will
            save memory when you do not need to
            preserve the contents of the input array.
            Treat the input as undefined, but it will
            probably be fully or partially sorted.
            Default is False. If overwrite_input
            is True and a is not already an ndarray,
            an error will be raised.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            A new array holding the result. If the input contains integers

        Examples
        --------
        >>> a = ivy.Array([[10.0, ivy.nan, 4], [3, 2, 1]])
        >>> a.nanmedian(a)
            3.0
        >>> a.nanmedian(a, axis=0)
            array([6.5, 2. , 2.5])
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
        """ivy.Array instance method variant of ivy.bincount. This method simply
        wraps the function, and so the docstring for ivy.bincount also applies to
        this method with minimal changes.

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
