# global
from typing import Optional, List, Union, Dict

# local
from ivy.container.base import ContainerBase
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithSorting(ContainerBase):
    @staticmethod
    def static_argsort(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.argsort. This method 
        simply wraps the function, and so the docstring for 
        ivy.argsort also applies to this method with minimal changes.
        
        Parameters
        ----------
        x 
            input array or container. Should have a numeric data type.
        axis
            axis along which to sort. If set to ``-1``, the function must sort 
            along the last axis. Default: ``-1``.
        descending
            sort order. If ``True``, the returned indices sort
            ``x`` in descending order (by value). If ``False``, 
            the returned indices sort ``x`` in ascending order 
            (by value). Default: ``False``.
        stable
            sort stability. If ``True``, the returned indices must maintain
            the relative order of ``x`` values which compare as equal.
            If ``False``, the returned indices may or may not maintain
            the relative order of ``x`` values which compare as equal (i.e., the 
            relative order of ``x`` values which compare as equal 
            is implementation-dependent). Default: ``True``.
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
            a container containing the index values of sorted
            array. The returned array must have a
            data type determined by :ref:`type-promotion`.
        
        Examples
        --------
        With: code:`ivy.Container` inputs:
        
        >>> x = ivy.Container(a=ivy.array([7, 2, 1]),\
                              b=ivy.array([3, 2]))
        >>> y = x.static_argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        {
            a: ivy.array([0, 1, 2]),
            b: ivy.array([0, 1])
        }
        
        >>> x = ivy.Container(a=ivy.array([7, 2, 1]),\
                              b=ivy.array([[3, 2], [7, 0.2]]))
        >>> y = x.static_argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        {
            a: ivy.array([0, 1, 2]),
            b: ivy.array([[0, 1]],[0, 1]])
        }
        
        With: code:`ivy.Container` inputs:
        
        >>> x = ivy.Container(a=ivy.array([2, 5, 1]),\
                              b=ivy.array([1, 5], [.2,.1]))
        >>> y = x.static_argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        {
            a: ivy.array([2, 0, 1]),
            b: ivy.array([[1, 0],\
                            [0,1]])
        }
        
        >>> x = ivy.Container(a=ivy.native_array([2, 5, 1]),\
                              b=ivy.array([1, 5], [.2,.1]))
        >>> y = x.static_argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        {
            a: ivy.array([2, 0, 1]),
            b: ivy.array([[1, 0],\
                            [0,1]])
        }
        
        """
        return ContainerBase.multi_map_in_static_method(
            "argsort",
            x,
            axis=axis,
            descending=descending,
            stable=stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def argsort(
        self: ivy.Container,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.argsort.
        This method simply wraps the function, and 
        so the docstring for ivy.argsort also applies to this method
        with minimal changes.
        
        Parameters
        ----------
        self 
            input array or container. Should have a numeric data type.
        axis
            axis along which to sort. If set to ``-1``, the function 
            must sort along the last axis. Default: ``-1``.
        descending
            sort order. If ``True``, the returned indices sort ``x``
            in descending order (by value). If ``False``, the 
            returned indices sort ``x`` in ascending order (by value).
            Default: ``False``.
        stable
            sort stability. If ``True``, the returned indices must
            maintain the relative order of ``x`` values which compare
            as equal. If ``False``, the returned indices may or may not
            maintain the relative order of ``x`` values which compare 
            as equal (i.e., the relative order of ``x`` values which 
            compare as equal is implementation-dependent).
            Default: ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). 
            Default is False.
        out
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.
            
        Returns
        -------
        ret 
            a container containing the index values of sorted array.
            The returned array must have a data type determined 
            by :ref:`type-promotion`.
        
        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([7, 2, 1]),\
                              b=ivy.array([3, 2]))
        >>> y = x.argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        {
            a: ivy.array([0, 1, 2]),
            b: ivy.array([0, 1])
        }
        """
        return self.static_argsort(
            self,
            axis=axis,
            descending=descending,
            stable=stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sort(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sort. This method simply wraps the
        function, and so the docstring for ivy.add also applies to this method
        with minimal changes.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([5, 9, 0.2]),\
                              b=ivy.array([[8, 1], [5, 0.8]]))
        >>> y = ivy.Container.static_sort(x)
        >>> print(y)
        {
            a: ivy.array([0.2, 5., 9.]),
            b: ivy.array([[1., 8.], [0.8, 5.]])
        }

        >>> x = ivy.Container(a=ivy.array([8, 0.5, 6]),\
                              b=ivy.array([[9, 0.7], [0.4, 0]]))
        >>> y = ivy.Container.static_sort(x)
        >>> print(y)
        {
            a: ivy.array([0.5, 6., 8.]),
            b: ivy.array([[0.7, 9.], [0., 0.4]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sort",
            x,
            axis=axis,
            descending=descending,
            stable=stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sort(
        self: ivy.Container,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sort. This method simply wraps the
        function, and so the docstring for ivy.sort also applies to this method
        with minimal changes.

        Examples
        --------
        With：code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([5, 9, 0.2]),\
                              b=ivy.array([8, 1]))
        >>> y = x.sort()
        >>> print(y)
        {
            a: ivy.array([0.2, 5., 9.]),
            b: ivy.array([1, 8])
        }

        >>> x = ivy.Container(a=ivy.array([5, 9, 0.2]), \
                              b=ivy.array([[8, 1], [5, 0.8]]))
        >>> y = x.sort()
        >>> print(y)
        { a: ivy.array([0.2, 5., 9.]), \
          b: ivy.array([[1., 8.],[0.8, 5.]])
        }

        With：code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([8, 0.5, 6]),\
                              b=ivy.array([[9, 0.7], [0.4, 0]]))
        >>> y = ivy.sort(x)
        >>> print(y)
        {
            a: ivy.array([0.5, 6., 8.]),
            b: ivy.array([[0.7, 9.],\
                            [0., 0.4]])
        }

        >>> x = ivy.Container(a=ivy.native_array([8, 0.5, 6]),\
                              b=ivy.array([[9, 0.7], [0.4, 0]]))
        >>> y = ivy.sort(x)
        >>> print(y)
        {
            a: ivy.array([0.5, 6., 8.]),
            b: ivy.array([[0.7, 9.],
                  [0., 0.4]])
        }

        """
        return self.static_sort(
            self,
            axis=axis,
            descending=descending,
            stable=stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_searchsorted(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        v: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        side="left",
        sorter=None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "searchsorted",
            x1,
            v,
            side=side,
            sorter=sorter,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def searchsorted(
        self: ivy.Container,
        v: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        side="left",
        sorter=None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_searchsorted(
            self,
            v,
            side=side,
            sorter=sorter,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
