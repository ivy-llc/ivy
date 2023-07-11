# global
from typing import Optional, Union, List, Dict, Tuple, Sequence

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithStatisticalExperimental(ContainerBase):
    @staticmethod
    def static_histogram(
        a: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        bins: Optional[
            Union[int, ivy.Array, ivy.NativeArray, ivy.Container, str]
        ] = None,
        axis: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        extend_lower_interval: Optional[bool] = False,
        extend_upper_interval: Optional[bool] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        range: Optional[Tuple[float]] = None,
        weights: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        density: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.<func_name>. This method simply wraps
        the function, and so the docstring for ivy.histogram also applies to this method
        with minimal changes.

        Parameters
        ----------
        a
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
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a tuple containing the values of the histogram and the bin edges.

        Both the description and the type hints above assumes an array input for
        simplicity, but this function is *nestable*, and therefore also accepts
        :class:`ivy.Container` instances in place of any of the arguments.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = ivy.array([0., 1., 2., 3., 4., 5.])
        >>> dtype = ivy.int32
        >>> z = ivy.Container.static_histogram(x, bins=y, dtype=dtype)
        >>> print(z.a)
        >>> print(z.b)
        (ivy.array([1, 1, 1, 0, 0]), ivy.array([0., 1., 2., 3., 4., 5.]))
        (ivy.array([0, 0, 0, 1, 2]), ivy.array([0., 1., 2., 3., 4., 5.]))
        """
        return ContainerBase.cont_multi_map_in_function(
            "histogram",
            a,
            bins=bins,
            axis=axis,
            extend_lower_interval=extend_lower_interval,
            extend_upper_interval=extend_upper_interval,
            dtype=dtype,
            range=range,
            weights=weights,
            density=density,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def histogram(
        self: ivy.Container,
        /,
        *,
        bins: Optional[
            Union[int, ivy.Array, ivy.NativeArray, ivy.Container, str]
        ] = None,
        axis: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        extend_lower_interval: Optional[bool] = False,
        extend_upper_interval: Optional[bool] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        range: Optional[Tuple[float]] = None,
        weights: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        density: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.<func_name>. This method simply
        wraps the function, and so the docstring for ivy.histogram also applies to this
        method with minimal changes.

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
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a tuple containing the values of the histogram and the bin edges.

        Both the description and the type hints above assumes an array input for
        simplicity, but this function is *nestable*, and therefore also accepts
        :class:`ivy.Container` instances in place of any of the arguments.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = ivy.array([0., 1., 2., 3., 4., 5.])
        >>> dtype = ivy.int32
        >>> z = ivy.histogram(x, bins=y, dtype=dtype)
        >>> print(z.a)
        >>> print(z.b)
        (ivy.array([1, 1, 1, 0, 0]), ivy.array([0., 1., 2., 3., 4., 5.]))
        (ivy.array([0, 0, 0, 1, 2]), ivy.array([0., 1., 2., 3., 4., 5.]))
        """
        return self.static_histogram(
            self,
            bins=bins,
            axis=axis,
            extend_lower_interval=extend_lower_interval,
            extend_upper_interval=extend_upper_interval,
            dtype=dtype,
            range=range,
            weights=weights,
            density=density,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_median(
        input: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.median. This method simply wraps the
        function, and so the docstring for ivy.median also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            Input container including arrays.
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
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.zeros((3, 4, 5)), b=ivy.zeros((2,7,6)))
        >>> ivy.Container.static_moveaxis(x, 0, -1).shape
        {
            a: (4, 5, 3)
            b: (7, 6, 2)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "median",
            input,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def median(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.median. This method simply wraps
        the function, and so the docstring for ivy.median also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
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
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(
        >>>     a=ivy.array([[10, 7, 4], [3, 2, 1]]),
        >>>     b=ivy.array([[1, 4, 2], [8, 7, 0]])
        >>> )
        >>> x.median(axis=0)
        {
            a: ivy.array([6.5, 4.5, 2.5]),
            b: ivy.array([4.5, 5.5, 1.])
        }
        """
        return self.static_median(self, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def static_nanmean(
        input: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.nanmean. This method simply wraps the
        function, and so the docstring for ivy.nanmean also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            Input container including arrays.
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
            The nanmean of the array elements in the container.

        Examples
        --------
        >>> a = ivy.Container(x=ivy.array([[1, ivy.nan], [3, 4]]),\
                                y=ivy.array([[ivy.nan, 1, 2], [1, 2, 3]])
        >>> ivy.Container.static_moveaxis(a)
        {
            x: 2.6666666666666665
            y: 1.8
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "nanmean",
            input,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def nanmean(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.nanmean. This method simply wraps
        the function, and so the docstring for ivy.nanmean also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
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
            The nanmean of the array elements in the input container.

        Examples
        --------
        >>> a = ivy.Container(x=ivy.array([[1, ivy.nan], [3, 4]]),\
                                y=ivy.array([[ivy.nan, 1, 2], [1, 2, 3]])
        >>> a.nanmean()
        {
            x: 2.6666666666666665
            y: 1.8
        }
        """
        return self.static_nanmean(
            self, axis=axis, keepdims=keepdims, dtype=dtype, out=out
        )

    @staticmethod
    def static_quantile(
        a: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        q: Union[ivy.Array, float],
        /,
        *,
        axis: Optional[Union[Sequence[int], int]] = None,
        keepdims: bool = False,
        interpolation: str = "linear",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.quantile. This method simply wraps
        the function, and so the docstring for ivy.quantile also applies to this method
        with minimal changes.

        Parameters
        ----------
        a
            Input container including arrays.
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
            Container with (rank(q) + N - len(axis)) dimensional arrays of same dtype
            as input arrays in the container, or, if axis is None, rank(q) arrays. The
            first rank(q) dimensions index quantiles for different values of q.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]),
                              y=ivy.array([1., 2., 3., 4.]))
        >>> q = 0.5
        >>> b = ivy.Container.static_quantile(a, q)
        >>> print(b)
        {
            x: 3.5,
            y: 2.5
        }

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]),
                              y=ivy.array([1., 2., 3., 4.]))
        >>> q = ivy.array([0.5, 0.75])
        >>> b = ivy.Container.static_quantile(a, q)
        >>> print(b)
        {
            x: ivy.array([3.5, 6.25]),
            y: ivy.array([2.5, 3.25])
        }

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]),
                              y=ivy.array([1., 2., 3., 4.]))
        >>> q = ivy.array([0.5, 0.75])
        >>> b = ivy.Container.static_quantile(a, q, axis = 0)
        >>> print(b)
        {
            x: ivy.array([[6.5, 4.5, 2.5],
                        [8.25, 5.75, 3.25]]),
            y: ivy.array([2.5, 3.25])
        }

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]))
        >>> b = ivy.Container.static_quantile(a, q, axis = 1, keepdims=True)
        >>> print(b)
        {
            x: ivy.array([[[7.],
                    [2.]],
                    [[8.5],
                    [2.5]]])
        }

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]),
                              y=ivy.array([1., 2., 3., 4.]))
        >>> q = ivy.array([0.3, 0.7])
        >>> b = ivy.Container.static_quantile(a, q, axis = 0, interpolation="lower")
        >>> print(b)
        {
            x: ivy.array([[3., 2., 1.],
                        [3., 2., 1.]]),
            y: ivy.array([1., 3.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "quantile",
            a,
            q,
            axis=axis,
            keepdims=keepdims,
            interpolation=interpolation,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def quantile(
        self: ivy.Container,
        q: Union[ivy.Array, float],
        /,
        *,
        axis: Optional[Union[Sequence[int], int]] = None,
        keepdims: bool = False,
        interpolation: str = "linear",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.quantile. This method simply wraps
        the function, and so the docstring for ivy.quantile also applies to this method
        with minimal changes.

        Parameters
        ----------
        a
            Input container including arrays.
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
            Container with (rank(q) + N - len(axis)) dimensional arrays of same dtype
            as input arrays in the container, or, if axis is None, rank(q) arrays. The 
            first rank(q) dimensions index quantiles for different values of q.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]),\
                              y=ivy.array([1., 2., 3., 4.]))
        >>> q = 0.5
        >>> b = a.quantile(q)
        >>> print(b)
        {
            x: 3.5,
            y: 2.5
        }

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]),
                              y=ivy.array([1., 2., 3., 4.]))
        >>> q = ivy.array([0.5, 0.75])
        >>> b = a.quantile(q)
        >>> print(b)
        {
            x: ivy.array([3.5, 6.25]),
            y: ivy.array([2.5, 3.25])
        }

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]),
                              y=ivy.array([1., 2., 3., 4.]))
        >>> q = ivy.array([0.5, 0.75])
        >>> b = a.quantile(q, axis = 0)
        >>> print(b)
        {
            x: ivy.array([[6.5, 4.5, 2.5], 
                        [8.25, 5.75, 3.25]]),
            y: ivy.array([2.5, 3.25])
        }

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]))
        >>> b = a.quantile(q, axis = 1, keepdims=True)
        >>> print(b)
        {
            x: ivy.array([[[7.], 
                    [2.]], 
                    [[8.5], 
                    [2.5]]])
        }

        >>> a = ivy.Container(x=ivy.array([[10., 7., 4.], [3., 2., 1.]]),
                              y=ivy.array([1., 2., 3., 4.]))
        >>> q = ivy.array([0.3, 0.7])
        >>> b = a.quantile(q, axis = 0, interpolation="lower")
        >>> print(b)
        {
            x: ivy.array([[3., 2., 1.],
                        [3., 2., 1.]]),
            y: ivy.array([1., 3.])
        }
        """
        return self.static_quantile(
            self,
            q,
            axis=axis,
            keepdims=keepdims,
            interpolation=interpolation,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_corrcoef(
        x: ivy.Container,
        /,
        *,
        y: Optional[ivy.Container] = None,
        rowvar: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = False,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.corrcoef. This method simply wraps
        the function, and so the docstring for ivy.corrcoef also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            Input container including arrays.
        y
            An additional input container.
        rowvar
            If rowvar is True (default), then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is transposed:
            each column represents a variable, while the rows contain observations.

        Returns
        -------
        ret
            The corrcoef of the array elements in the container.

        Examples
        --------
        >>> a = ivy.Container(w=ivy.array([[1., 2.], [3., 4.]]), \
                                 z=ivy.array([[0., 1., 2.], [2., 1., 0.]]))
        >>> ivy.Container.corrcoef(a)
        {
            w: ivy.array([[1., 1.], 
                          [1., 1.]]),
            z: ivy.array([[1., -1.], 
                          [-1., 1.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "corrcoef",
            x,
            y=y,
            rowvar=rowvar,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def corrcoef(
        self: ivy.Container,
        /,
        *,
        y: Optional[ivy.Container] = None,
        rowvar: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.corrcoef. This method simply wraps
        the function, and so the docstring for ivy.corrcoef also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
        y
            An additional input container.
        rowvar
            If rowvar is True (default), then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is transposed:
            each column represents a variable, while the rows contain observations.

        Returns
        -------
        ret
            The corrcoef of the array elements in the input container.

        Examples
        --------
        >>> a = ivy.Container(w=ivy.array([[1., 2.], [3., 4.]]), \
                                 z=ivy.array([[0., 1., 2.], [2., 1., 0.]]))
        >>> ivy.Container.corrcoef(a)
        {
            w: ivy.array([[1., 1.], 
                          [1., 1.]]),
            z: ivy.array([[1., -1.], 
                          [-1., 1.]])
        }
        """
        return self.static_corrcoef(self, y=y, rowvar=rowvar, out=out)

    @staticmethod
    def static_nanmedian(
        input: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        overwrite_input: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.median. This method simply wraps the
        function, and so the docstring for ivy.median also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            Input container including arrays.
        axis
            Axis or axes along which the medians are computed. The default is to compute
            the median along a flattened version of the array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one.
        overwrite_input
            If True, then allow use of memory of input array for calculations.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The median of the array elements.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([[10.0, ivy.nan, 4], [3, 2, 1]]))
        >>> ivy.Container.static_nanmedian(x)
        {
            a: ivy.array(3.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "nanmedian",
            input,
            axis=axis,
            keepdims=keepdims,
            overwrite_input=overwrite_input,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def nanmedian(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
        overwrite_input: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.nanmedian. This method simply wraps
        the function, and so the docstring for ivy.nanmedian also applies to this method
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
            correctly against the original container. If the value is anything
            but the default, then keepdims will be passed through to the mean or
            sum methods of sub-classes of ndarray. If the sub-classes methods
            does not implement keepdims any exceptions will be raised.
        overwrite_input
            If True, then allow use of memory of input array a for calculations.
            The input array will be modified by the call to median.
            This will save memory when you do not need to preserve
            the contents of the input array.Treat the input as undefined,
            but it will probably be fully or partially sorted.
            Default is False. If overwrite_input is True and
            input container does not already have leaves which are
            of the ndarray kind, an error will be raised.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            A new array holding the result. If the input contains integers

        Examples
        --------
        With :class:`ivy.Container` input and default backend set as `numpy`:
        >>> x = ivy.Container(a=ivy.array([[10.0, ivy.nan, 4], [3, 2, 1]]),
                b=ivy.array([[12, 10, 34], [45, 23, ivy.nan]]))
        >>> x.nanmedian()
        {
            a: ivy.array(3.),
            b: ivy.array(23.)
        }
        >>> x.nanmedian(axis=0)
        {
            a: ivy.array([6.5, 2., 2.5]),
            b: ivy.array([28.5, 16.5, 34.])
        }
        """
        return self.static_nanmedian(
            self, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
        )

    @staticmethod
    def static_bincount(
        x: ivy.Container,
        /,
        *,
        weights: Optional[ivy.Container] = None,
        minlength: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.bincount. This method simply wraps
        the function, and so the docstring for ivy.bincount also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            Input container including arrays.
        weights
            An optional input container including arrays.
        minlength
            A minimum number of bins for the output array.

        Returns
        -------
        ret
            The bincount of the array elements.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1, 1, 2, 2, 2, 3]),
                            b=ivy.array([1, 1, 2, 2, 2, 3]))
        >>> ivy.Container.static_bincount(x)
            {
                a: array([0, 2, 3, 1])
                b: array([0, 2, 3, 1])
            }
        """
        return ContainerBase.cont_multi_map_in_function(
            "bincount",
            x,
            weights=weights,
            minlength=minlength,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bincount(
        self: ivy.Container,
        /,
        *,
        weights: Optional[ivy.Container] = None,
        minlength: int = 0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Array instance method variant of ivy.bincount. This method simply wraps the
        function, and so the docstring for ivy.bincount also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        weights
            An optional input array.
        minlength
            A minimum number of bins for the output array.

        Returns
        -------
        ret
            The bincount of the array elements.

        Examples
        --------
        >>> a = ivy.Container([[10.0, ivy.nan, 4], [3, 2, 1]])
        >>> a.bincount(a)
            3.0
        >>> a.bincount(a, axis=0)
            array([6.5, 2. , 2.5])
        """
        return self.static_bincount(self, weights=weights, minlength=minlength, out=out)

    @staticmethod
    def static_igamma(
        a: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.igamma. This method simply wraps the
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
        return ContainerBase.cont_multi_map_in_function(
            "igamma",
            a,
            x=x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def igamma(
        self: ivy.Container,
        /,
        *,
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.igamma. This method simply wraps
        the function, and so the docstring for ivy.igamma also applies to this method
        with minimal changes.

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
        return self.static_igamma(self, x=x, out=out)

    @staticmethod
    def static_cov(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container] = None,
        /,
        *,
        rowVar: bool = True,
        bias: bool = False,
        ddof: int = None,
        fweights: ivy.Array = None,
        aweights: ivy.Array = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cov. This method simply wraps the
        function, and so the docstring for ivy.cov also applies to this method with
        minimal changes.

        Parameters
        ----------
        x1
            a 1D or 2D input array, nativearray or container, with a numeric data type.
        x2
            optional second 1D or 2D input array, nativearray, or container, with a
            numeric data type. Must have the same shape as x1.
        rowVar
            optional variable where each row of input is interpreted as a variable
            (default = True). If set to False, each column is instead interpreted
            as a variable.
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
            will have at least ``numpy.float64`` precision.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the covariance matrix of an input matrix, or the
            covariance matrix of two variables. The returned container must have a
            floating-point data type determined by Type Promotion Rules and must be
            a square matrix of shape (N, N), where N is the number of rows in the
            input(s).

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.array([1., 2., 3.])
        >>> y = ivy.Container(a=ivy.array([3. ,2. ,1.]), b=ivy.array([-1., -2., -3.]))
        >>> z = ivy.Container.static_cov(x, y)
        >>> print(z)
        {
            a: ivy.array([ 1., -1.]
                         [-1.,  1.]),
            b: ivy.array([ 1., -1.]
                         [-1.,  1.])
        }

        With multiple :class:`ivy.Container` inputs:
        >>> x = ivy.Container(a=ivy.array([1., 2., 3.]), b=ivy.array([1., 2., 3.]))
        >>> y = ivy.Container(a=ivy.array([3., 2., 1.]), b=ivy.array([3., 2., 1.]))
        >>> z = ivy.Container.static_cov(x, y)
        >>> print(z)
        {
            a: ivy.container([ 1., -1., -1., -1.]
                         [ 1.,  1., -1., -1.]),
            b: ivy.container([-1., -1.,  1.,  1.]
                         [-1.,  1.,  1.,  1.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cov",
            x1,
            x2,
            rowVar=rowVar,
            bias=bias,
            ddof=ddof,
            fweights=fweights,
            aweights=aweights,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def cov(
        self: ivy.Container,
        x2: ivy.Container = None,
        /,
        *,
        rowVar: bool = True,
        bias: bool = False,
        ddof: Optional[int] = None,
        fweights: Optional[ivy.Array] = None,
        aweights: Optional[ivy.Array] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cov. This method simply wraps the
        function, and so the docstring for ivy.cov also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            a 1D or 2D input container, with a numeric data type.
        x2
            optional second 1D or 2D input array, nativearray, or container, with a
            numeric data type. Must have the same shape as ``self``.
        rowVar
            optional variable where each row of input is interpreted as a variable
            (default = True). If set to False, each column is instead interpreted
            as a variable.
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
            will have at least ``numpy.float64`` precision.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the covariance matrix of an input matrix, or the
            covariance matrix of two variables. The returned container must have a
            floating-point data type determined by Type Promotion Rules and must be
            a square matrix of shape (N, N), where N is the number of variables in the
            input(s).

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., 2., 3.]), b=ivy.array([1., 2., 3.]))
        >>> y = ivy.Container(a=ivy.array([3., 2., 1.]), b=ivy.array([3., 2., 1.]))
        >>> z = x.cov(y)
        >>> print(z)
        {
            a: ivy.container([ 1., -1., -1., -1.]
                         [ 1.,  1., -1., -1.]),
            b: ivy.container([-1., -1.,  1.,  1.]
                         [-1.,  1.,  1.,  1.])
        }
        """
        return self.static_cov(
            self,
            x2,
            rowVar=rowVar,
            bias=bias,
            ddof=ddof,
            fweights=fweights,
            aweights=aweights,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def cummax(
        self: ivy.Container,
        /,
        *,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cummax. This method simply wraps
        the function, and so the docstring for ivy.cummax also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container to cummax at leaves.
        axis
            Axis along which the cumulative product is computed. Default is ``0``.
        exclusive
            Whether to exclude the first element of the input array.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        dtype
            Data type of the returned array. Default is ``None``.
        out
            Optional output container. Default is ``None``.

        Returns
        -------
        ret
            Containers with arrays cummax at leaves along specified axis.

        --------
        With one :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = x.cummax(axis=0)
        >>> print(y)
        {
            a: ivy.array([1, 2, 3]),
            b: ivy.array([4, 5, 6])
        }

        >>> x = ivy.Container(a=ivy.array([[2, 3], [5, 7], [11, 13]]),
                              b=ivy.array([[3, 4], [4, 5], [5, 6]]))
        >>> y = ivy.Container(a = ivy.zeros((3, 2)), b = ivy.zeros((3, 2)))
        >>> x.cummax(axis=1, exclusive=True, out=y)
        {
            a: ivy.array([[2., 3.],
                          [5., 7.],
                          [11., 13.]]),
            b: ivy.array([[3., 4.],
                          [4., 5.],
                          [5., 6.]])
        }
        """
        return self._static_cummax(
            self,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def cummin(
        self: ivy.Container,
        /,
        *,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cummin. This method simply wraps
        the function, and so the docstring for ivy.cummin also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container to cummin at leaves.
        axis
            Axis along which the cumulative product is computed. Default is ``0``.
        exclusive
            Whether to exclude the first element of the input array.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        dtype
            Data type of the returned array. Default is ``None``.
        out
            Optional output container. Default is ``None``.

        Returns
        -------
        ret
            Containers with arrays cummin at leaves along specified axis.

        Examples #TODO: change examples and change doc string
        --------
        With one :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = x.cummin(axis=0)
        >>> print(y)
        {
            a: ivy.array([1, 1, 1]),
            b: ivy.array([4, 4, 4])
        }

        >>> x = ivy.Container(a=ivy.array([[2, 3], [5, 7], [11, 13]]),
                              b=ivy.array([[3, 4], [4, 5], [5, 6]]))
        >>> y = ivy.Container(a = ivy.zeros((3, 2)), b = ivy.zeros((3, 2)))
        >>> x.cummin(axis=1, out=y)
        {
            a: ivy.array([[2, 2],
                          [5, 5],
                          [11, 11]]),
            b: ivy.array([[3, 3],
                          [4, 4],
                          [5, 5]])
        }
        """
        return self._static_cummin(
            self,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def _static_cummax(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cummax. This method simply wraps the
        function, and so the docstring for ivy.cummax also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input array or container to cummax.
        axis
            Axis to cummax along. Default is ``0``.
        exclusive
            Whether to exclude the first element of the input array.
            Default is ``False``.
        reverse
            Whether to perform the cummax from last to first element in the selected
            axis. Default is ``False`` (from first to last element)
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            Optional output container. Default is ``None``.

        Returns
        -------
        ret
            Containers with arrays cummax at leaves along specified axis.

        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container.static_cummax(x, axis=0)
        >>> print(y)
        {
            a: ivy.array([1, 2, 3]),
            b: ivy.array([4, 5, 6])
        }

        >>> x = ivy.Container(a=ivy.array([[2, 3], [5, 7], [11, 13]]),
                              b=ivy.array([[3, 4], [4, 5], [5, 6]]))
        >>> y = ivy.Container(a = ivy.zeros((3, 2)), b = ivy.zeros((3, 2)))
        >>> ivy.Container.static_cummax(x, axis=1, out=y)
        >>> print(y)
        {
            a: ivy.array([[2., 3.],
                          [5., 7.],
                          [11., 13.]]),
            b: ivy.array([[3., 4.],
                          [4., 5.],
                          [5., 6.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cummax",
            x,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def _static_cummin(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cummin. This method simply wraps the
        function, and so the docstring for ivy.cummin also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input array or container to cummin.
        axis
            Axis to cummin along. Default is ``0``.
        exclusive
            Whether to exclude the first element of the input array.
            Default is ``False``.
        reverse
            Whether to perform the cummin from last to first element in the selected
            axis. Default is ``False`` (from first to last element)
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        dtype
            Data type of the returned array. Default is ``None``.
        out
            Optional output container. Default is ``None``.

        Returns
        -------
        ret
            Containers with arrays cummin at leaves along specified axis.

        Examples #TODO: fix examples and this doc
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container.static_cummin(x, axis=0)
        >>> print(y)
        {
            a: ivy.array([1, 1, 1]),
            b: ivy.array([4, 4, 4])
        }

        >>> x = ivy.Container(a=ivy.array([[2, 3], [5, 7], [11, 13]]),
                              b=ivy.array([[3, 4], [4, 5], [5, 6]]))
        >>> y = ivy.Container(a = ivy.zeros((3, 2)), b = ivy.zeros((3, 2)))
        >>> x.static_cummin(axis=1, out=y)
        {
            a: ivy.array([[2, 2],
                          [5, 5],
                          [11, 11]]),
            b: ivy.array([[3, 3],
                          [4, 4],
                          [5, 5]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cummin",
            x,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )
