# global
from typing import Optional, List, Union
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class _ArrayWithNorms(abc.ABC):
    def layer_norm(
        self: ivy.Array,
        normalized_idxs: List[int],
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
        normalized_idxs
            Indices to apply the normalization to.
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

        Examples
        --------
        >>> x = ivy.array([[0.0976, -0.3452,  1.2740],
        ...                   [0.1047,  0.5886,  1.2732],
        ...                   [0.7696, -1.7024, -2.2518]])
        >>> norm = x.layer_norm([0, 1], eps=0.001,
        ...                     new_std=1.5, scale=0.5, offset=[0.5, 0.02, 0.1])
        >>> print(norm)
        ivy.array([[ 0.826, -0.178, 0.981 ],
                   [ 0.831,  0.421, 0.981 ],
                   [ 1.26 , -1.05 , -1.28 ]])
        """
        return ivy.layer_norm(
            self,
            normalized_idxs,
            scale=scale,
            offset=offset,
            eps=eps,
            new_std=new_std,
            out=out,
        )
