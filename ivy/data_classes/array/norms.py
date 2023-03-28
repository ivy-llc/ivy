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
        scale: Optional[Union[ivy.Array, float]] = None,
        b: Optional[Union[ivy.Array, float]] = None,
        epsilon: float = ivy._MIN_BASE,
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
        b
            Learnable beta variables for elementwise post-addition, default is ``None``.
        epsilon
            small constant to add to the denominator, use global ivy._MIN_BASE by
            default.
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
        >>> norm = x.layer_norm([0, 1], epsilon=0.001,
        ...                     new_std=1.5, scale=0.5, b=[0.5, 0.02, 0.1])
        >>> print(norm)
        ivy.array([[ 0.826, -0.178, 0.981 ],
                   [ 0.831,  0.421, 0.981 ],
                   [ 1.26 , -1.05 , -1.28 ]])

        """
        return ivy.layer_norm(
            self,
            normalized_idxs,
            scale=scale,
            b=b,
            epsilon=epsilon,
            new_std=new_std,
            out=out,
        )
