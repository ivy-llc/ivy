# global
import abc
from typing import Optional, Union, Sequence

# local
import ivy


class _ArrayWithTensors(abc.ABC):
    def unfold(
        self: ivy.Array, /, mode: Optional[int] = 0, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.unfold. This method simply wraps the
        function, and so the docstring for ivy.unfold also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            input tensor to be unfolded
        mode
            indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

        Returns
        -------
        ret
            unfolded_tensor of shape ``(tensor.shape[mode], -1)``
        """
        return ivy.unfold(self._data, mode=mode, out=out)

    def fold(
        self: ivy.Array,
        /,
        mode: int,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.fold. This method simply wraps the
        function, and so the docstring for ivy.fold also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            unfolded tensor of shape ``(shape[mode], -1)``
        mode
            the mode of the unfolding
        shape
            shape of the original tensor before unfolding

        Returns
        -------
        ret
            unfolded_tensor of shape ``(tensor.shape[mode], -1)``
        """
        return ivy.fold(self._data, mode=mode, shape=shape, out=out)
