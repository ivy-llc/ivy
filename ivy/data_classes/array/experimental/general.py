# global
import abc
from typing import Union, Callable, Sequence

# local
import ivy


class _ArrayWithGeneralExperimental(abc.ABC):
    def reduce(
        self: ivy.Array,
        init_value: Union[int, float],
        computation: Callable,
        /,
        *,
        axes: Union[int, Sequence[int]] = 0,
        keepdims: bool = False,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.reduce. This method simply
        wraps the function, and so the docstring for ivy.reduce also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The array to act on.
        init_value
            The value with which to start the reduction.
        computation
            The reduction function.
        axes
            The dimensions along which the reduction is performed.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one.

        Returns
        -------
        ret
            The reduced array.

        Examples
        --------
        >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
        >>> x.reduce(0, ivy.add, 0)
        ivy.array([6, 15])
        """
        return ivy.reduce(self, init_value, computation, axes=axes, keepdims=keepdims)
