# global
from typing import Optional, List, Union

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class _ContainerWithNorms(ContainerBase):
    def layer_norm(
        self: ivy.Container,
        normalized_idxs: List[int],
        /,
        *,
        scale: Optional[Union[ivy.Array, float]] = None,
        b: Optional[Union[ivy.Array, float]] = None,
        epsilon: float = ivy._MIN_BASE,
        new_std: float = 1.0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.layer_norm. This method simply
        wraps the function, and so the docstring for ivy.layer_norm also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input container
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The layer after applying layer normalization.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]),
        ...                    'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
        >>> normalized_idxs = [0]
        >>> norm = x.layer_norm(normalized_idxs, epsilon=1.25, scale=0.3)
        >>> print(norm)
        {
            a: ivy.array([-0.342, 0.0427, 0.299]),
            b: ivy.array([[-0.241, -0.241, -0.241,
                          [0.241, 0.241, 0.241]])
        }

        With multiple :class:`ivy.Container` inputs:

        >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]),
        ...                    'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
        >>> normalized_idxs = ivy.Container({'a': [0], 'b': [1]})
        >>> new_std = ivy.Container({'a': 1.25, 'b': 1.5})
        >>> bias = ivy.Container({'a': [0.2, 0.5, 0.7], 'b': 0.3})
        >>> norm = x.layer_norm(normalized_idxs, new_std=new_std, b=b)
        >>> print(norm)
        {
            a: ivy.array([-1.62, 0.203, 1.42]),
            b: ivy.array([[-1.84, 0., 1.84],
                          [-1.84, 0., 1.84]])
        }

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
