# global
from typing import Optional, Union, List, Dict, Tuple, Sequence

# local
import ivy
from ivy.container.base import ContainerBase


class _ContainerWithStatisticalExperimental(ContainerBase):
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
        ivy.Container static method variant of ivy.median. This method simply wraps
        the function, and so the docstring for ivy.median also applies to this method
        with minimal changes.

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
        """ivy.Container instance method variant of ivy.median. This method simply
        wraps the function, and so the docstring for ivy.median also applies to this
        method with minimal changes.

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
        ivy.Container static method variant of ivy.nanmean. This method simply wraps
        the function, and so the docstring for ivy.nanmean also applies to this method
        with minimal changes.

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
        """ivy.Container instance method variant of ivy.nanmean. This method simply
        wraps the function, and so the docstring for ivy.nanmean also applies to this
        method with minimal changes.

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
    def static_unravel_index(
        indices: ivy.Container,
        shape: Tuple[int],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.unravel_index.
        This method simply wraps the function, and so the docstring
        for ivy.unravel_index also applies to this method with minimal
        changes.

        Parameters
        ----------
        indices
            Input container including arrays.
        shape
            The shape of the array to use for unraveling indices.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Container with tuples that have arrays with the same shape as
            the arrays in the input container.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> indices = ivy.Container(a=ivy.array([22, 41, 37])), b=ivy.array([30, 2]))
        >>> ivy.Container.static_unravel_index(indices, (7,6))
        {
            a: (ivy.array([3, 6, 6]), ivy.array([4, 5, 1]))
            b: (ivy.array([5, 0], ivy.array([0, 2])))
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "unravel_index",
            indices,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def unravel_index(
        self: ivy.Container,
        shape: Tuple[int],
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.unravel_index.
        This method simply wraps the function, and so the docstring for
        ivy.unravel_index also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
        shape
            The shape of the array to use for unraveling indices.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Container with tuples that have arrays with the same shape as
            the arrays in the input container.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> indices = ivy.Container(a=ivy.array([22, 41, 37])), b=ivy.array([30, 2]))
        >>> indices.unravel_index((7, 6))
        {
            a: (ivy.array([3, 6, 6]), ivy.array([4, 5, 1]))
            b: (ivy.array([5, 0], ivy.array([0, 2])))
        }
        """
        return self.static_unravel_index(self, shape, out=out)

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
        """ivy.Container static method variant of ivy.quantile.
        This method simply wraps the function, and so the docstring
        for ivy.quantile also applies to this method with minimal
        changes.

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
        """ivy.Container instance method variant of ivy.quantile.
        This method simply wraps the function, and so the docstring
        for ivy.quantile also applies to this method with minimal
        changes.

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
        """ivy.Container instance method variant of ivy.corrcoef. This method simply
        wraps the function, and so the docstring for ivy.corrcoef also applies to this
        method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.median. This method simply wraps
        the function, and so the docstring for ivy.median also applies to this method
        with minimal changes.

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
        >>> ivy.Container.static_nanmedian(x, 0, -1).shape
        {
            a: (4, 5, 3)
            b: (7, 6, 2)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "nanmedian",
            input,
            axis=axis,
            keepdims=keepdims,
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
        overwrite_input: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
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
            The input array will be modified by the call to median.
            This will save memory when you do not need to preserve
            the contents of the input array.Treat the input as undefined,
            but it will probably be fully or partially sorted.
            Default is False. If overwrite_input is True and
            a is not already an ndarray,
            an error will be raised.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            A new array holding the result. If the input contains integers

        Examples
        --------
        >>> a = ivy.Container([[10.0, ivy.nan, 4], [3, 2, 1]])
        >>> a.nanmedian(a)
            3.0
        >>> a.nanmedian(a, axis=0)
            array([6.5, 2. , 2.5])
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
        """ivy.Array instance method variant of ivy.bincount. This method simply
        wraps the function, and so the docstring for ivy.bincount also applies to
        this method with minimal changes.

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
