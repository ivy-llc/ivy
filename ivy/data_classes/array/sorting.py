# global
import abc
from typing import Optional, Union, Literal, List

# local

import ivy


class _ArrayWithSorting(abc.ABC):
    def argsort(
        self: ivy.Array,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.argsort. This method simply
        wraps the function, and so the docstring for ivy.argsort also applies
        to this method with minimal changes.

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
        out
            optional output array, for writing the result to. It must have the same
            shape as input.

        Returns
        -------
        ret
            an array of indices. The returned array must have the same shape as ``x``.
            The returned array must have the default array index data type.

        Examples
        --------
        >>> x = ivy.array([1, 5, 2])
        >>> y = x.argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        ivy.array([1, 2, 0])

        >>> x = ivy.array([9.6, 2.7, 5.2])
        >>> y = x.argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        ivy.array([0, 2, 1])
        """
        return ivy.argsort(
            self._data, axis=axis, descending=descending, stable=stable, out=out
        )

    def sort(
        self: ivy.Array,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.sort. This method simply
        wraps the function, and so the docstring for ivy.sort also applies to
        this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([7, 8, 6])
        >>> y = x.sort(axis=-1, descending=True, stable=False)
        >>> print(y)
        ivy.array([8, 7, 6])

        >>> x = ivy.array([8.5, 8.2, 7.6])
        >>> y = x.sort(axis=-1, descending=True, stable=False)
        >>> print(y)
        ivy.array([8.5, 8.2, 7.6])
        """
        return ivy.sort(
            self._data, axis=axis, descending=descending, stable=stable, out=out
        )

    def msort(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.msort. This method simply
        wraps the function, and so the docstring for ivy.msort also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            sorted array of the same type and shape as a

        Examples
        --------
        >>> a = ivy.asarray([[8, 9, 6],[6, 2, 6]])
        >>> a.msort()
        ivy.array(
            [[6, 2, 6],
            [8, 9, 6]]
            )
        """
        return ivy.msort(self._data, out=out)

    def searchsorted(
        self: ivy.Array,
        v: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: Optional[Union[ivy.Array, ivy.NativeArray, List[int]]] = None,
        ret_dtype: Union[ivy.Dtype, ivy.NativeDtype] = ivy.int64,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.searchsorted.

        This method simply wraps the function, and so the docstring for
        ivy.searchsorted also applies to this method with minimal
        changes.
        """
        return ivy.searchsorted(
            self.data, v, side=side, sorter=sorter, ret_dtype=ret_dtype, out=out
        )
