# global
import abc
from typing import Optional, Union

# local

import ivy


class ArrayWithSorting(abc.ABC):
    def argsort(
        self: ivy.Array,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.argsort. This method simply wraps the
        function, and so the docstring for ivy.argsort also applies to this method
        with minimal changes.
        
        Parameters
        ----------
        self 
            input array.
        axis
            axis along which to sort. If set to ``-1``, the function
            must sort along the last axis. Default: ``-1``.
        descending
            sort order. If ``True``, the returned indices sort ``x`` in descending order 
            (by value). If ``False``, the returned indices sort ``x`` in ascending order 
            (by value). Default: ``False``.
        stable
            sort stability. If ``True``, the returned indices
            must maintain the relative order of ``x`` values 
            which compare as equal. If ``False``, the returned
            indices may or may not maintain the relative order
            of ``x`` values which compare as equal (i.e., the
            relative order of ``x`` values which compare as 
            equal is implementation-dependent). Default: ``True``.
        
        Returns
        -------
        out 
            an array of indices. The returned array must have the same shape as ``x``.
            The returned array must have the default array index data type.
        
        Examples
        --------
        >>> x = ivy.array([1, 5, 2])
        >>> y = x.argsort(-1, True, False)
        >>> print(y)
        ivy.array([1, 2, 0])
        
        >>> x = ivy.array([9.6, 2.7, 5.2])
        >>> y = x.argsort(-1, True, False)
        >>> print(y)
        ivy.array([0, 2, 1])
        
        """
        return ivy.argsort(self._data, axis, descending, stable, out=out)
       
    def sort(
        self: ivy.Array,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sort. This method simply wraps the
        function, and so the docstring for ivy.sort also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([7, 8, 6])
        >>> y = x.sort(-1, True, False)
        >>> print(y)
        ivy.array([8, 7, 6])

        >>> x = ivy.array([8.5, 8.2, 7.6])
        >>> y = x.sort(-1, True, False)
        >>> print(y)
        ivy.array([8.5, 8.2, 7.6])

        """
        return ivy.sort(self._data, axis, descending, stable, out=out)
    
    def searchsorted(  
        self: ivy.Array,
        v: Union[ivy.Array, ivy.NativeArray], 
        side="left",
        sorter=None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.searchsorted(self.data, v, side=side, sorter=None, out=out)
