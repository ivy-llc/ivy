# global
import abc
from typing import Optional, Tuple


import ivy


class _ArrayWithSet(abc.ABC):
    def unique_counts(self: ivy.Array) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_counts. This method simply wraps
        the function, and so the docstring for ivy.unique_counts also applies to this
        method with minimal changes.

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
        """Return the unique elements of an input array `x`.
        .. admonition:: Data-dependent output shape
            :class: important
            The shapes of two of the output arrays for this function depend on the
            data values in the input array; hence, array libraries which build
            computation graphs (e.g., JAX, Dask, etc.) may find this function
            difficult to implement without knowing array values. Accordingly,
            such libraries may choose to omit this function.
            See :ref:`data-dependent-output-shapes` section for more details.
        .. note::
            Uniqueness should be determined based on value equality
            (i.e., ``x_i == x_j``). For input arrays having floating-point
            data types, value-based equality implies the following behavior.
            -   As ``nan`` values compare as ``False``, ``nan`` values
                should be considered distinct.
            -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should
                not be considered distinct, and the corresponding unique
                element will be implementation-dependent (e.g., an
                implementation could choose to return ``-0`` if ``-0`` occurs
                before ``+0``).

        Parameters
        ----------
        x : ivy.Array or ivy.NativeArray
            Input array. If `x` has more than one dimension, the function must flatten
            `x` and return the unique elements of the flattened array.
        out : ivy.Array, optional
            Optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ivy.Array
            An array containing the set of unique elements in `x`. The returned
            array must have the same data type as `x`.
            .. note::
                The order of unique elements is not specified and may vary
                between implementations.

        Raises
        ------
        TypeError
            If `x` is not an instance of `ivy.Array` or `ivy.NativeArray`.

        Examples
        --------
        >>> import ivy
        >>> x = ivy.array([1, 2, 2, 3, 4, 4, 4])
        >>> print(x.unique_values())
        ivy.array([1, 2, 3, 4])

        >>> x = ivy.array([[1, 2], [3, 4]])
        >>> print(x.unique_values())
        ivy.array([1, 2, 3, 4])
        """
        return ivy.unique_values(self._data, out=out)

    def unique_all(
        self: ivy.Array,
        /,
        *,
        axis: Optional[int] = None,
        by_value: bool = True,
    ) -> Tuple[ivy.Array, ivy.Array, ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_all. This method simply wraps
        the function, and so the docstring for ivy.unique_all also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array.

        axis
            the axis to apply unique on. If None, the unique elements of the flattened
            ``x`` are returned.

        by_value
            If False, the unique elements will be sorted in the same order that they
            occur in ''x''. Otherwise, they will be sorted by value.

        Returns
        -------
        ret
            a namedtuple ``(values, indices, inverse_indices, counts)``.
            The details can be found in the docstring for ivy.unique_all.


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
        return ivy.unique_all(self._data, axis=axis, by_value=by_value)

    def unique_inverse(self: ivy.Array) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_inverse. This method simply
        wraps the function, and so the docstring for ivy.unique_inverse also applies to
        this method with minimal changes.

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
