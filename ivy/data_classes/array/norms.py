# global
from typing import Optional, Tuple, Union
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class _ArrayWithNorms(abc.ABC):
    def layer_norm(
        self: ivy.Array,
        normalized_shape: Tuple[int],
        /,
        *,
        scale: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        offset: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        eps: float = 1e-05,
        new_std: float = 1.0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.layer_norm. This method simply wraps
        the function, and so the docstring for ivy.layer_norm also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array
        normalized_shape
            Tuple containing the last k dimensions to apply normalization to.
        scale
            Learnable gamma variables for elementwise post-multiplication,
            default is ``None``.
        offset
            Learnable beta variables for elementwise post-addition, default is ``None``.
        eps
            small constant to add to the denominator. Default is ``1e-05``.
        new_std
            The standard deviation of the new normalized values. Default is 1.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The layer after applying layer normalization.

        """
        return ivy.layer_norm(
            self,
            normalized_shape,
            scale=scale,
            offset=offset,
            eps=eps,
            new_std=new_std,
            out=out,
        )
