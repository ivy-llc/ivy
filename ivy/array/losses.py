# global
import abc
from typing import Optional, Union

# local
import ivy


class ArrayWithLosses(abc.ABC):
    def cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        epsilon: float = 1e-7,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.cross_entropy. This method
        simply wraps the function, and so the docstring for ivy.cross_entropy
        also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0, 0, 1, 0])
        >>> y = ivy.array([0.25, 0.25, 0.25, 0.25])
        >>> z = x.cross_entropy(y)
        >>> print(z)
        ivy.array(1.3862944)
        """
        return ivy.cross_entropy(self._data, pred, axis=axis, epsilon=epsilon, out=out)

    def binary_cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        epsilon: float = 1e-7,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.binary_cross_entropy. This method
        simply wraps the function, and so the docstring for ivy.binary_cross_entropy
        also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([1 , 1, 0])
        >>> y = ivy.array([0.7, 0.8, 0.2])
        >>> z = x.binary_cross_entropy(y)
        >>> print(z)
        ivy.array([0.357, 0.223, 0.223])
        """
        return ivy.binary_cross_entropy(self._data, pred, epsilon=epsilon, out=out)

    def sparse_cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        epsilon: float = 1e-7,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sparse_cross_entropy. This method
        simply wraps the function, and so the docstring for ivy.sparse_cross_entropy
        also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([1 , 1, 0])
        >>> y = ivy.array([0.7, 0.8, 0.2])
        >>> z = x.sparse_cross_entropy(y)
        >>> print(z)
        ivy.array([0.223, 0.223, 0.357])
        """
        return ivy.sparse_cross_entropy(
            self._data, pred, axis=axis, epsilon=epsilon, out=out
        )
