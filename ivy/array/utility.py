# global
from typing import Optional, Union, Sequence
import abc

# local
import ivy


class ArrayWithUtility(abc.ABC):
    def all(
        self: ivy.Array,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.all. This method simply wraps the
        function, and so the docstring for ivy.all also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0, 1, 2])
        >>> y = x.all()
        >>> print(y)
        ivy.array(False)
        """
        return ivy.all(self._data, axis, keepdims, out=out)

    def any(
        self: ivy.Array,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.any. This method simply wraps the
        function, and so the docstring for ivy.any also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0, 1, 2])
        >>> y = x.any()
        >>> print(y)
        ivy.array(True)
        """
        return ivy.any(self._data, axis, keepdims, out=out)
