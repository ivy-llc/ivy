# global
from typing import Optional, Union, Dict, Sequence

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithUtility(ContainerBase):
    @staticmethod
    def static_all(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.all. This method simply wraps the
        function, and so the docstring for ivy.all also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            axis or axes along which to perform a logical AND reduction. By default, a
            logical AND reduction must be performed over the entire array. If a tuple of
            integers, logical AND reductions must be performed over multiple axes. A
            valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N``
            is the rank(number of dimensions) of ``self``. If an ``axis`` is specified
            as a negative integer, the function must determine the axis along which to
            perform a reduction by counting backward from the last dimension (where
            ``-1`` refers to the last dimension). If provided an invalid ``axis``, the
            function must raise an exception. Default  ``None``.
        keepdims
            If ``True``, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the
            reduced axes(dimensions) must not be included in the result.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            if a logical AND reduction was performed over the entire array, the returned
            container must be a zero-dimensional array containing the test result;
            otherwise, the returned container must be a non-zero-dimensional array
            containing the test results. The returned container must have a data type of
            ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([0, 1, 1]))
        >>> y = ivy.Container.static_all(x)
        >>> print(y)
        {
            a: ivy.array(False),
            b: ivy.array(False)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "all",
            x,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def all(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.all. This method simply wraps the
        function, and so the docstring for ivy.all also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            axis or axes along which to perform a logical AND reduction. By default, a
            logical AND reduction must be performed over the entire array. If a tuple of
            integers, logical AND reductions must be performed over multiple axes. A
            valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N``
            is the rank(number of dimensions) of ``self``. If an ``axis`` is specified
            as a negative integer, the function must determine the axis along which to
            perform a reduction by counting backward from the last dimension (where
            ``-1`` refers to the last dimension). If provided an invalid ``axis``, the
            function must raise an exception. Default  ``None``.
        keepdims
            If ``True``, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the
            reduced axes(dimensions) must not be included in the result.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            if a logical AND reduction was performed over the entire array, the returned
            container must be a zero-dimensional array containing the test result;
            otherwise, the returned container must have non-zero-dimensional arrays
            containing the test results. The returned container must have a data type of
            ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([0, 1, 1]))
        >>> y = x.all()
        >>> print(y)
        {
            a: ivy.array(False),
            b: ivy.array(False)
        }
        """
        return self.static_all(
            self,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_any(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.any. This method simply wraps the
        function, and so the docstring for ivy.any also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            axis or axes along which to perform a logical OR reduction. By default, a
            logical OR reduction must be performed over the entire array. If a tuple of
            integers, logical OR reductions must be performed over multiple axes. A
            valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N``
            is the rank(number of dimensions) of ``self``. If an ``axis`` is specified
            as a negative integer, the function must determine the axis along which to
            perform a reduction by counting backward from the last dimension (where
            ``-1`` refers to the last dimension). If provided an invalid ``axis``, the
            function must raise an exception. Default: ``None``.
        keepdims
            If ``True``, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the
            reduced axes(dimensions) must not be included in the result.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            if a logical OR reduction was performed over the entire array, the returned
            container must be a zero-dimensional array containing the test result;
            otherwise, the returned container must have non-zero-dimensional arrays
            containing the test results. The returned container must have a data type of
            ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([0, 1, 1]))
        >>> y = ivy.Container.static_any(x)
        >>> print(y)
        {
            a: ivy.array(True),
            b: ivy.array(True)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "any",
            x,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def any(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.any. This method simply wraps the
        function, and so the docstring for ivy.any also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            axis or axes along which to perform a logical OR reduction. By default, a
            logical OR reduction must be performed over the entire array. If a tuple of
            integers, logical OR reductions must be performed over multiple axes. A
            valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N``
            is the rank(number of dimensions) of ``self``. If an ``axis`` is specified
            as a negative integer, the function must determine the axis along which to
            perform a reduction by counting backward from the last dimension (where
            ``-1`` refers to the last dimension). If provided an invalid ``axis``, the
            function must raise an exception. Default: ``None``.
        keepdims
            If ``True``, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the
            reduced axes(dimensions) must not be included in the result.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            if a logical OR reduction was performed over the entire array, the returned
            container must be a zero-dimensional array containing the test result;
            otherwise, the returned container must have non-zero-dimensional arrays
            containing the test results. The returned container must have a data type of
            ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([0, 1, 1]))
        >>> y = x.any()
        >>> print(y)
        {
            a: ivy.array(True),
            b: ivy.array(True)
        }
        """
        return self.static_any(
            self,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
