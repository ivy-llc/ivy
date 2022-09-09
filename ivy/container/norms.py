# global
from typing import Optional, List

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithNorms(ContainerBase):
    def layer_norm(
        self: ivy.Container,
        normalized_idxs: List[int],
        /,
        *,
        epsilon: float = ivy._MIN_BASE,
        scale: float = 1.0,
        offset: float = 1.0,
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The layer after applying layer normalization.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                               'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
        >>> normalized_idxs = [0]
        >>> norm = x.layer_norm(normalized_idxs, epsilon=1.25, scale=0.3)
        >>> print(norm)
        {
            a: ivy.array([0.658, 1.04, 1.3]),
            b: ivy.array([[0.759, 0.759, 0.759], 
                          [1.24, 1.24, 1.24]])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                               'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
        >>> normalized_idxs = ivy.Container({'a': [0], 'b': [1]})
        >>> new_std = ivy.Container({'a': 1.25, 'b': 1.5})
        >>> offset = ivy.Container({'a': 0.2, 'b': 0.3})
        >>> norm = x.layer_norm(normalized_idxs, new_std=new_std, offset=offset)
        >>> print(norm)
        {
            a: ivy.array([-1.42, 0.403, 1.62]),
            b: ivy.array([[-1.54, 0.3, 2.14], 
                          [-1.54, 0.3, 2.14]])
        }

        """
        return ivy.layer_norm(
            self,
            normalized_idxs,
            epsilon=epsilon,
            scale=scale,
            offset=offset,
            new_std=new_std,
            out=out,
        )
