# global
from typing import Optional, Tuple, Union

# local
import ivy
from ivy.data_classes.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class _ContainerWithNorms(ContainerBase):
    def layer_norm(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        normalized_shape: Tuple[int],
        /,
        *,
        scale: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        offset: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        eps: float = 1e-05,
        new_std: float = 1.0,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.layer_norm. This method simply
        wraps the function, and so the docstring for ivy.layer_norm also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input container
        normalized_shape
            Tuple containing the last k dimensions to apply normalization to.
        scale
            Learnable gamma variables for elementwise post-multiplication,
            default is ``None``.
        offset
            Learnable beta variables for elementwise post-addition, default is ``None``.
        epsilon
            small constant to add to the denominator. Default is ``1e-05``.
        new_std
            The standard deviation of the new normalized values. Default is 1.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

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
