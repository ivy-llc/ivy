"""Ivy wrapping functions for conversions.

Collection of Ivy functions for wrapping functions to accept and return
ivy.Array instances.
"""

# global
from typing import Union, Dict, Optional, List

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithConversions(ContainerBase):
    @staticmethod
    def _static_to_native(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        nested: Union[bool, ivy.Container] = False,
        include_derived: Optional[Union[Dict[str, bool], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.to_native.

        This method simply wraps the function, and so the docstring for ivy.to_native
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to be converted.
        nested
            Whether to apply the conversion on arguments in a nested manner. If so, all
            dicts, lists and tuples will be traversed to their lowest leaves in search
            of ivy.Array instances. Default is ``False``.
        include_derived
            Whether to also recursive for classes derived from tuple, list and dict.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container object with all sub-arrays converted to their native format.
        """
        return ContainerBase.cont_multi_map_in_function(
            "to_native",
            x,
            nested=nested,
            include_derived=include_derived,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def to_native(
        self: ivy.Container,
        nested: Union[bool, ivy.Container] = False,
        include_derived: Optional[Union[Dict[str, bool], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.to_native.

        This method simply wraps the function, and so the docstring for ivy.to_native
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to be converted.
        nested
            Whether to apply the conversion on arguments in a nested manner. If so, all
            dicts, lists and tuples will be traversed to their lowest leaves in search
            of ivy.Array instances. Default is ``False``.
        include_derived
            Whether to also recursive for classes derived from tuple, list and dict.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container object with all sub-arrays converted to their native format.
        """
        return self._static_to_native(
            self,
            nested,
            include_derived,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def _static_to_ivy(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        nested: Union[bool, ivy.Container] = False,
        include_derived: Optional[Union[Dict[str, bool], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.to_ivy.

        This method simply wraps the function, and so the docstring for ivy.to_ivy also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to be converted.
        nested
            Whether to apply the conversion on arguments in a nested manner. If so, all
            dicts, lists and tuples will be traversed to their lowest leaves in search
            of ivy.Array instances. Default is ``False``.
        include_derived
            Whether to also recursive for classes derived from tuple, list and dict.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container object with all native sub-arrays converted to their ivy.Array
            instances.
        """
        return ContainerBase.cont_multi_map_in_function(
            "to_ivy",
            x,
            nested=nested,
            include_derived=include_derived,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def to_ivy(
        self: ivy.Container,
        nested: Union[bool, ivy.Container] = False,
        include_derived: Optional[Union[Dict[str, bool], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.to_ivy.

        This method simply wraps the function, and so the docstring for ivy.to_ivy also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to be converted.
        nested
            Whether to apply the conversion on arguments in a nested manner. If so,
            all dicts, lists and tuples will be traversed to their lowest leaves in
            search of ivy.Array instances. Default is ``False``.
        include_derived
            Whether to also recursive for classes derived from tuple, list and dict.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container object with all native sub-arrays converted to their ivy.Array
            instances.
        """
        return self._static_to_ivy(
            self,
            nested,
            include_derived,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
