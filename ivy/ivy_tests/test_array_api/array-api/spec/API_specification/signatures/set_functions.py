from ._types import Tuple, array

def unique_all(x: array, /) -> Tuple[array, array, array, array]:
    """
    Returns the unique elements of an input array ``x``, the first occurring indices for each unique element in ``x``, the indices from the set of unique elements that reconstruct ``x``, and the corresponding counts for each unique element in ``x``.

    .. admonition:: Data-dependent output shape
        :class: important

        The shapes of two of the output arrays for this function depend on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

    .. note::
       Uniqueness should be determined based on value equality (i.e., ``x_i == x_j``). For input arrays having floating-point data types, value-based equality implies the following behavior.

       -   As ``nan`` values compare as ``False``, ``nan`` values should be considered distinct.

       -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return ``-0`` if ``-0`` occurs before ``+0``).

       As signed zeros are not distinct, using ``inverse_indices`` to reconstruct the input array is not guaranteed to return an array having the exact same values.

       Each ``nan`` value should have a count of one, while the counts for signed zeros should be aggregated as a single count.

    Parameters
    ----------
    x: array
        input array. If ``x`` has more than one dimension, the function must flatten ``x`` and return the unique elements of the flattened array.

    Returns
    -------
    out: Tuple[array, array, array, array]
        a namedtuple ``(values, indices, inverse_indices, counts)`` whose

        - first element must have the field name ``values`` and must be an array containing the unique elements of ``x``. The array must have the same data type as ``x``.
        - second element must have the field name ``indices`` and must be an array containing the indices (first occurrences) of ``x`` that result in ``values``. The array must have the same shape as ``values`` and must have the default array index data type.
        - third element must have the field name ``inverse_indices`` and must be an array containing the indices of ``values`` that reconstruct ``x``. The array must have the same shape as ``x`` and must have the default array index data type.
        - fourth element must have the field name ``counts`` and must be an array containing the number of times each unique element occurs in ``x``. The returned array must have same shape as ``values`` and must have the default array index data type.

        .. note::
           The order of unique elements is not specified and may vary between implementations.
    """

def unique_counts(x: array, /) -> Tuple[array, array]:
    """
    Returns the unique elements of an input array ``x`` and the corresponding counts for each unique element in ``x``.

    .. admonition:: Data-dependent output shape
        :class: important

        The shapes of two of the output arrays for this function depend on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

    .. note::
       Uniqueness should be determined based on value equality (i.e., ``x_i == x_j``). For input arrays having floating-point data types, value-based equality implies the following behavior.

       -   As ``nan`` values compare as ``False``, ``nan`` values should be considered distinct.
       -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return ``-0`` if ``-0`` occurs before ``+0``).

       Each ``nan`` value should have a count of one, while the counts for signed zeros should be aggregated as a single count.

    Parameters
    ----------
    x: array
        input array. If ``x`` has more than one dimension, the function must flatten ``x`` and return the unique elements of the flattened array.

    Returns
    -------
    out: Tuple[array, array]
        a namedtuple `(values, counts)` whose

        -   first element must have the field name ``values`` and must be an array containing the unique elements of ``x``. The array must have the same data type as ``x``.
        -   second element must have the field name `counts` and must be an array containing the number of times each unique element occurs in ``x``. The returned array must have same shape as ``values`` and must have the default array index data type.

        .. note::
           The order of unique elements is not specified and may vary between implementations.
    """

def unique_inverse(x: array, /) -> Tuple[array, array]:
    """
    Returns the unique elements of an input array ``x`` and the indices from the set of unique elements that reconstruct ``x``.

    .. admonition:: Data-dependent output shape
        :class: important

        The shapes of two of the output arrays for this function depend on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

    .. note::
       Uniqueness should be determined based on value equality (i.e., ``x_i == x_j``). For input arrays having floating-point data types, value-based equality implies the following behavior.

       -   As ``nan`` values compare as ``False``, ``nan`` values should be considered distinct.
       -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return ``-0`` if ``-0`` occurs before ``+0``).

       As signed zeros are not distinct, using ``inverse_indices`` to reconstruct the input array is not guaranteed to return an array having the exact same values.

    Parameters
    ----------
    x: array
        input array. If ``x`` has more than one dimension, the function must flatten ``x`` and return the unique elements of the flattened array.

    Returns
    -------
    out: Tuple[array, array]
        a namedtuple ``(values, inverse_indices)`` whose

        -   first element must have the field name ``values`` and must be an array containing the unique elements of ``x``. The array must have the same data type as ``x``.
        -   second element must have the field name ``inverse_indices`` and must be an array containing the indices of ``values`` that reconstruct ``x``. The array must have the same shape as ``x`` and have the default array index data type.

        .. note::
           The order of unique elements is not specified and may vary between implementations.
    """

def unique_values(x: array, /) -> array:
    """
    Returns the unique elements of an input array ``x``.

    .. admonition:: Data-dependent output shape
        :class: important

        The shapes of two of the output arrays for this function depend on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

    .. note::
       Uniqueness should be determined based on value equality (i.e., ``x_i == x_j``). For input arrays having floating-point data types, value-based equality implies the following behavior.

       -   As ``nan`` values compare as ``False``, ``nan`` values should be considered distinct.
       -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return ``-0`` if ``-0`` occurs before ``+0``).

    Parameters
    ----------
    x: array
        input array. If ``x`` has more than one dimension, the function must flatten ``x`` and return the unique elements of the flattened array.

    Returns
    -------
    out: array
        an array containing the set of unique elements in ``x``. The returned array must have the same data type as ``x``.

        .. note::
           The order of unique elements is not specified and may vary between implementations.
    """

__all__ = ['unique_all', 'unique_counts', 'unique_inverse', 'unique_values']