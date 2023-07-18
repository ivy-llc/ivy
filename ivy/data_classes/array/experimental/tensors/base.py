# global
import abc
from typing import Optional

# local
import ivy


class _ArrayWithTensors(abc.ABC):
    def tensor_to_vec(
        self: ivy.Array, /, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tensor_to_vec. This method simply wraps
        the function, and so the docstring for ivy.tensor_to_vec also applies to this
        method with minimal changes.

        Parameters
        ----------
        input
            input tensor of shape ``(i_1, ..., i_n)``

        Returns
        -------
        ret
            1D-array vectorised tensor of shape ``(i_1 * i_2 * ... * i_n)``
        """
        return ivy.tensor_to_vec(self._data, out=out)
