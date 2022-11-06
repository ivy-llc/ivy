# global
import abc
from typing import Optional, Union

# local
import ivy


class ArrayWithLinalgExperimental(abc.ABC):
    def diagflat(
        self: Union[ivy.Array, ivy.NativeArray],
        *,
        offset: Optional[int] = 0,
        padding_value: Optional[float] = 0,
        align: Optional[str] = "RIGHT_LEFT",
        num_rows: Optional[int] = -1,
        num_cols: Optional[int] = -1,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.diagflat.
        This method simply wraps the function, and so the docstring for
        ivy.diagflat also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([1,2])
        >>> x.diagflat(k=1)
        ivy.array([[0, 1, 0],
                   [0, 0, 2],
                   [0, 0, 0]])
        """
        return ivy.diagflat(
            self._data,
            offset=offset,
            padding_value=padding_value,
            align=align,
            num_rows=num_rows,
            num_cols=num_cols,
            out=out,
        )

    def kron(
        self: ivy.Array,
        b: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.kron.
        This method simply wraps the function, and so the docstring for
        ivy.kron also applies to this method with minimal changes.

        Examples
        --------
        >>> a = ivy.array([1,2])
        >>> b = ivy.array([3,4])
        >>> a.diagflat(b)
        ivy.array([3, 4, 6, 8])
        """
        return ivy.kron(self._data, b, out=out)
