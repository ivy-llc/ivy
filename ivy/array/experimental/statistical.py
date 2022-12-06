# global
import abc
from typing import Optional, Union, Tuple, Sequence

# local
import ivy


class ArrayWithStatisticalExperimental(abc.ABC):
    def median(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: Optional[bool] = False,
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
        keepdims: Optional[bool] = False,
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
        a
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
