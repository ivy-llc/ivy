# global
from typing import Dict, List, Optional, Union

# local
from ivy.container.base import ContainerBase
import ivy


class ContainerWithSet(ContainerBase):
    @staticmethod
    def static_unique_all(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_static_method(
            "unique_all",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def unique_all(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_unique_all(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_unique_counts(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.unique_counts. This method simply
        wraps the function, and so the docstring for ivy.unique_counts also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input container. If ``x`` has more than one dimension, the function must
            flatten ``x`` and return the unique elements of the flattened array.
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

        Returns
        -------
        ret
            a namedtuple ``(values, counts)`` whose

            - first element must have the field name ``values`` and must be an
            array containing the unique elements of ``x``.
            The array must have the same data type as ``x``.
            - second element must have the field name ``counts`` and must be an array
            containing the number of times each unique element occurs in ``x``.
            The returned array must have same shape as ``values`` and must
            have the default array index data type.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 3. , 2. , 1. , 0.]),
        ...                   b=ivy.array([1,2,1,3,4,1,3]))
        >>> y = ivy.static_unique_counts(x)
        >>> print(y)
        {
            a:[values=ivy.array([0.,1.,2.,3.]),counts=ivy.array([2,2,1,1])],
            b:[values=ivy.array([1,2,3,4]),counts=ivy.array([3,1,2,1])]
        }
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "unique_counts",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def unique_counts(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.unique_counts. This method
        simply wraps the function, and so the docstring for ivy.unique_counts
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. If ``x`` has more than one dimension, the function must
            flatten ``x`` and return the unique elements of the flattened array.
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

        Returns
        -------
        ret
            a namedtuple ``(values, counts)`` whose

            - first element must have the field name ``values`` and must be an
            array containing the unique elements of ``x``.
            The array must have the same data type as ``x``.
            - second element must have the field name ``counts`` and must be an array
            containing the number of times each unique element occurs in ``x``.
            The returned array must have same shape as ``values`` and must
            have the default array index data type.

        Examples
        --------
        With :class:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([0., 1., 3. , 2. , 1. , 0.]),
        ...                   b=ivy.array([1,2,1,3,4,1,3]))
        >>> y = x.unique_counts()
        >>> print(y)
        {
            a:[values=ivy.array([0.,1.,2.,3.]),counts=ivy.array([2,2,1,1])],
            b:[values=ivy.array([1,2,3,4]),counts=ivy.array([3,1,2,1])]}
        """
        return self.static_unique_counts(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_unique_values(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_static_method(
            "unique_values",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def unique_values(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_unique_values(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_unique_inverse(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_static_method(
            "unique_inverse",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def unique_inverse(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.unique_inverse. This method simply
        wraps the function, and so the docstring for ivy.unique_inverse also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
             input container. If ``x`` has more than one dimension, the function must
             flatten ``x`` and return the unique elements of the flattened array.
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

        Returns
        -------
        ret

             a namedtuple ``(values, inverse_indices)`` whose

             - first element must have the field name ``values`` and must be an array
             containing the unique elements of ``x``. The array must have the same data
             type as ``x``.
             - second element must have the field name ``inverse_indices`` and
              must be an array containing the indices of ``values`` that
              reconstruct ``x``. The array must have the same shape as ``x`` and
              must have the default array index data type.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([4.,8.,3.,5.,9.,4.]),
        ...                   b=ivy.array([7,6,4,5,6,3,2]))
        >>> y = x.unique_inverse()
        >>> print(y)
        {
            a:[values=ivy.array([3.,4.,5.,8.,9.]),inverse_indices=ivy.array([1,3,0,2,4,1])],
            b:[values=ivy.array([2,3,4,5,6,7]),inverse_indices=ivy.array([5,4,2,3,4,1,0])]}

        """
        return self.static_unique_inverse(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
