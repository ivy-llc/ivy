# global
import abc
from typing import Optional, Tuple


import ivy


class _ArrayWithSet(abc.ABC):
    def unique_counts(self: ivy.Array) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_counts. This method simply
        wraps the function, and so the docstring for ivy.unique_counts also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. If ``x`` has more than one dimension, the function must flatten
            ``x`` and return the unique elements of the flattened array.

        Returns
        -------
        ret
            a namedtuple ``(values, counts)`` whose

            - first element must have the field name ``values`` and must be an
            array containing the unique elements of ``x``.
            The array must have the same data type as ``x``.
            - second element must have the field name ``counts`` and must be an array
            containing the number of times each unique element occurs in ``x``.
            The returned array must have same shape as ``values`` and must
            have the default array index data type.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2. , 1. , 0.])
        >>> y = x.unique_counts()
        >>> print(y)
        Results(values=ivy.array([0.,1.,2.]),counts=ivy.array([2,2,1]))
        """
        return ivy.unique_counts(self._data)

    def unique_values(
        self: ivy.Array, /, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.unique_values(self._data, out=out)

    def unique_all(
        self: ivy.Array,
    ) -> Tuple[ivy.Array, ivy.Array, ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_all. This method simply
        wraps the function, and so the docstring for ivy.unique_all also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. If ``self`` has more than one dimension, the function
            must flatten ``self`` and return the unique elements of the
            flattened array.

        Returns
        -------
        ret
            a namedtuple ``(values, indices, inverse_indices, counts)``.
            The details can be found in the docstring for ivy.unique_all.

            .. note::
               The order of unique elements is not specified and may vary between
               implementations.

        Examples
        --------
        >>> x = ivy.randint(0, 10, shape=(2, 2), seed=0)
        >>> z = x.unique_all()
        >>> print(z)
        Results(values=ivy.array([1, 2, 5, 9]),
                indices=ivy.array([3, 2, 1, 0]),
                inverse_indices=ivy.array([[3, 2], [1, 0]]),
               counts=ivy.array([1, 1, 1, 1]))

        """
        return ivy.unique_all(self._data)

    def unique_inverse(self: ivy.Array) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_inverse. This method simply
        wraps the function, and so the docstring for ivy.unique_inverse also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. If ``x`` has more than one dimension, the function must
            flatten ``x`` and return the unique elements of the flattened array.

        Returns
        -------
        ret

            a namedtuple ``(values, inverse_indices)`` whose

            - first element must have the field name ``values`` and must be an array
              containing the unique elements of ``x``. The array must have the same data
              type as ``x``.
            - second element must have the field name ``inverse_indices`` and must be
              an array containing the indices of ``values`` that reconstruct ``x``.
              The array must have the same shape as ``x`` and must have the default
              array index data type.

        Examples
        --------
        >>> x = ivy.array([0.3,0.4,0.7,0.4,0.2,0.8,0.5])
        >>> y = x.unique_inverse()
        >>> print(y)
        Results(values=ivy.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.8]),
                inverse_indices=ivy.array([1, 2, 4, 2, 0, 5, 3]))

        """
        return ivy.unique_inverse(self._data)
