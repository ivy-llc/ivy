# global
from typing import Optional, List
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithNorms(abc.ABC):
    def layer_norm(
        self: ivy.Array,
        normalized_idxs: List[int],
        /,
        *,
        epsilon: float = ivy._MIN_BASE,
        scale: float = None,
        offset: float = None,
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
        epsilon
            small constant to add to the denominator, use global ivy._MIN_BASE by
            default.
        scale
            Learnable gamma variables for post-multiplication, default is None.
        offset
            Learnable beta variables for post-addition, default is None.
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
        >>> x = ivy.array([[0.0976, -0.3452,  1.2740], \
                           [0.1047,  0.5886,  1.2732], \
                           [0.7696, -1.7024, -2.2518]])
        >>> norm = x.layer_norm([0, 1], epsilon=0.001, \
                                new_std=1.5, offset=0.5, scale=0.5)
        >>> print(norm)
        ivy.array([[ 0.576,  0.292,  1.33 ],
                   [ 0.581,  0.891,  1.33 ],
                   [ 1.01 , -0.579, -0.931]])

        """
        return ivy.layer_norm(
            self, normalized_idxs, epsilon, scale, offset, new_std, out=out
        )
