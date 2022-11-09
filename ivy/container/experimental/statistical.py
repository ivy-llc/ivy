# global
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
)

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithStatisticalExperimental(ContainerBase):
    @staticmethod
    def static_median(
        input: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: Optional[bool] = False,
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
        return ContainerBase.multi_map_in_static_method(
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
        keepdims: Optional[bool] = False,
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
        keepdims: Optional[bool] = False,
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
        return ContainerBase.multi_map_in_static_method(
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
        keepdims: Optional[bool] = False,
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
