# global
from typing import Union, Tuple, Optional

# local
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions


# Array API Standard #
# -------------------#


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
def unique_all(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    by_value: bool = True,
) -> Tuple[
    Union[ivy.Array, ivy.NativeArray],
    Union[ivy.Array, ivy.NativeArray],
    Union[ivy.Array, ivy.NativeArray],
    Union[ivy.Array, ivy.NativeArray],
]:
    """Return the unique elements of an input array ``x``, the first occurring indices
    for each unique element in ``x``, the indices from the set of unique elements that
    reconstruct ``x``, and the corresponding counts for each unique element in ``x``.

    .. admonition:: Data-dependent output shape
        :class: important

        The shapes of two of the output arrays for this function depend on the data
        values in the input array; hence, array libraries which build computation graphs
        (e.g., JAX, Dask, etc.) may find this function difficult to implement without
        knowing array values. Accordingly, such libraries may choose to omit this
        function. See :ref:`data-dependent-output-shapes` section for more details.

    .. note::
       Uniqueness should be determined based on value equality (i.e., ``x_i == x_j``).
       For input arrays having floating-point data types, value-based equality implies
       the following behavior.

       -   As ``nan`` values compare as ``False``, ``nan`` values should be considered
           distinct.

       -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be
           considered distinct, and the corresponding unique element will be
           implementation-dependent (e.g., an implementation could choose to return
           ``-0`` if ``-0`` occurs before ``+0``).

       As signed zeros are not distinct, using ``inverse_indices`` to reconstruct the
       input array is not guaranteed to return an array having the exact same values.

       Each ``nan`` value should have a count of one, while the counts for signed zeros
       should be aggregated as a single count.

    Parameters
    ----------
    x
        input array.

    axis
        the axis to apply unique on. If None, the unique elements of the flattened ``x``
        are returned.

    by_value
        If False, the unique elements will be sorted in the same order that they occur
        in ''x''. Otherwise, they will be sorted by value.

    Returns
    -------
    ret
        a namedtuple ``(values, indices, inverse_indices, counts)`` whose
        - first element must have the field name ``values`` and must be an array
          containing the unique elements of ``x``. The array must have the same data
          type as ``x``.
        - second element must have the field name ``indices`` and must be an array
          containing the indices (first occurrences) of ``x`` that result in ``values``.
          The array must have the same length as ``values`` and must have the default
          array index data type.
        - third element must have the field name ``inverse_indices`` and must be an
          array containing the indices of ``values`` that reconstruct ``x``. The array
          must have the same length as the ``axis`` dimension of ``x`` and must have the
          default array index data type.
        - fourth element must have the field name ``counts`` and must be an array
          containing the number of times each unique element occurs in ``x``. The
          returned array must have the same length as ``values`` and must have the
          default array index data type.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.unique_all.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.randint(0, 10, shape=(2, 2), seed=0)
    >>> z = ivy.unique_all(x)
    >>> print(z)
    Results(values=ivy.array([1, 2, 5, 9]),
            indices=ivy.array([3, 2, 1, 0]),
            inverse_indices=ivy.array([[3, 2], [1, 0]]),
           counts=ivy.array([1, 1, 1, 1]))

    >>> x = ivy.array([[ 2.1141,  0.8101,  0.9298,  0.8460],
    ...                       [-1.2119, -0.3519, -0.6252,  0.4033],
    ...                       [ 0.7443,  0.2577, -0.3707, -0.0545],
    ...                       [-0.3238,  0.5944,  0.0775, -0.4327]])
    >>> x[range(4), range(4)] = ivy.nan #Introduce NaN values
    >>> z = ivy.unique_all(x)
    >>> print(z)
    Results(values=ivy.array([-1.2119    , -0.62519997, -0.3238    , -0.0545    ,
        0.0775    ,    0.2577    ,  0.40329999,  0.59439999,  0.74430001,  0.81010002,
        0.84600002,  0.92979997,         nan,         nan,         nan,         nan]),
        indices=ivy.array([ 4,  6, 12, 11, 14,  9,  7, 13,  8,  1,  3,  2,  0,  5,
                            10, 15]),
        inverse_indices=ivy.array([[12,  9, 11, 10],
                                   [ 0, 12,  1,  6],
                                   [ 8,  5, 12,  3],
                                   [ 2,  7,  4, 12]]),
       counts=ivy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    """
    return ivy.current_backend(x).unique_all(x, axis=axis, by_value=by_value)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
def unique_inverse(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> Tuple[Union[ivy.Array, ivy.NativeArray], Union[ivy.Array, ivy.NativeArray]]:
    """Return the unique elements of an input array ``x``, and the indices from the
     set of unique elements that reconstruct ``x``.

     .. admonition:: Data-dependent output shape
        :class: important

        The shapes of two of the output arrays for this function depend on the data
        values in the input array; hence, array libraries which build computation graphs
        (e.g., JAX, Dask, etc.) may find this function difficult to implement without
        knowing array values. Accordingly, such libraries may choose to omit this
        function. See :ref:`data-dependent-output-shapes` section for more details.

     .. note::
       Uniqueness should be determined based on value equality (i.e., ``x_i == x_j``).
       For input arrays having floating-point data types, value-based equality implies
       the following behavior.

       -   As ``nan`` values compare as ``False``, ``nan`` values should be considered
           distinct.

       -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be
           considered distinct, and the corresponding unique element will be
           implementation-dependent (e.g., an implementation could choose to return
           ``-0`` if ``-0`` occurs before ``+0``).

       As signed zeros are not distinct, using ``inverse_indices`` to reconstruct the
       input array is not guaranteed to return an array having the exact same values.


    Parameters
    ----------
    x
        input array. If ``x`` has more than one dimension, the function must flatten
        ``x`` and return the unique elements of the flattened array.

    Returns
    -------
    ret

        a namedtuple ``(values, inverse_indices)`` whose
        - first element must have the field name ``values`` and must be an array
          containing the unique elements of ``x``. The array must have the same data
          type as ``x``.
        - second element must have the field name ``inverse_indices`` and must be an
          array containing the indices of ``values`` that reconstruct ``x``. The array
          must have the same shape as ``x`` and must have the default array index data
          type.

        .. note::
           The order of unique elements is not specified and may vary between
           implementations.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.unique_inverse.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([4,5,3,2,4,1,3])
    >>> y = ivy.unique_inverse(x)
    >>> print(y)
    Results(values=ivy.array([1, 2, 3, 4, 5]),
            inverse_indices=ivy.array([3, 4, 2, 1, 3, 0, 2]))

    >>> x = ivy.array([0.5,0.3,0.8,0.2,1.2,2.4,0.3])
    >>> y = ivy.ivy.unique_inverse(x)
    >>> print(y)
    Results(values=ivy.array([0.2, 0.3, 0.5, 0.8, 1.2, 2.4]),
            inverse_indices=ivy.array([2, 1, 3, 0, 4, 5, 1]))

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1., 4., 3. , 5. , 3. , 7.]),
    ...                   b=ivy.array([3, 2, 6, 3, 7, 4, 9]))
    >>> y = ivy.ivy.unique_inverse(x)
    >>> print(y)
    {
        a: (list[2], <class ivy.array.array.Array> shape=[5]),
        b: (list[2], <class ivy.array.array.Array> shape=[6])
    }
    """
    return ivy.current_backend(x).unique_inverse(x)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def unique_values(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the unique elements of an input array ``x``.

    .. admonition:: Data-dependent output shape
        :class: important

        The shapes of two of the output arrays for this function depend on the data
        values in the input array; hence, array libraries which build computation graphs
        (e.g., JAX, Dask, etc.) may find this function difficult to implement without
        knowing array values. Accordingly, such libraries may choose to omit this
        function. See :ref:`data-dependent-output-shapes` section for more details.

    .. note::
       Uniqueness should be determined based on value equality (i.e., ``x_i == x_j``).
       For input arrays having floating-point data types, value-based equality implies
       the following behavior.

       -   As ``nan`` values compare as ``False``, ``nan`` values should be considered
           distinct.
       -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be
           considered distinct, and the corresponding unique element will be
           implementation-dependent (e.g., an implementation could choose to return
           ``-0`` if ``-0`` occurs before ``+0``).

    Parameters
    ----------
    x
        input array. If ``x`` has more than one dimension, the function must flatten
        ``x`` and return the unique elements of the flattened array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the set of unique elements in ``x``. The returned array must
        have the same data type as ``x``.

        .. note::
           The order of unique elements is not specified and may vary between
           implementations.

    Raises
    ------
    TypeError
        If `x` is not an instance of `ivy.Array` or `ivy.NativeArray`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.unique_values.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> import ivy
    >>> a = ivy.array([1, 1, 2, 2, 3, 4, 4, 5])
    >>> ivy.unique_values(a)
    array([1, 2, 3, 4, 5])

    >>> b = ivy.array([1, 2, 3, 4, 5])
    >>> ivy.unique_values(b)
    array([1, 2, 3, 4, 5])

    >>> c = ivy.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, -0.0, 0.0, float('nan'),
    ...                float('nan')])
    >>> ivy.unique_values(c)
    array([0., 1., 2., 3., 4., 5., nan, -0.])

    """
    return ivy.current_backend(x).unique_values(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
def unique_counts(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> Tuple[Union[ivy.Array, ivy.NativeArray], Union[ivy.Array, ivy.NativeArray]]:
    """
    Return the unique elements of an input array ``x`` and the corresponding counts for
    each unique element in ``x``.

    .. admonition:: Data-dependent output shape
        :class: important

        The shapes of two of the output arrays for this function depend on the data
        values in the input array; hence, array libraries which build computation graphs
        (e.g., JAX, Dask, etc.) may find this function difficult to implement without
        knowing array values. Accordingly, such libraries may choose to omit this
        function. See :ref:`data-dependent-output-shapes` section for more details.

    .. note::
       Uniqueness should be determined based on value equality (i.e., ``x_i == x_j``).
       For input arrays having floating-point data types, value-based equality implies
       the following behavior.

       -   As ``nan`` values compare as ``False``, ``nan`` values should be considered
           distinct.
       -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be
           considered distinct, and the corresponding unique element will be
           implementation-dependent (e.g., an implementation could choose to return
           ``-0`` if ``-0`` occurs before ``+0``).

    Parameters
    ----------
    x
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

    .. note::
           The order of unique elements is not specified and may vary between
           implementations.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.unique_counts.htmll>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1,2,1,3,4,1,3])
    >>> y = ivy.unique_counts(x)
    >>> print(y)
    Results(values=ivy.array([1, 2, 3, 4]), counts=ivy.array([3, 1, 2, 1]))

    >>> x = ivy.asarray([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
    >>> y = ivy.unique_counts(x)
    >>> print(y)
    Results(values=ivy.array([1, 2, 3, 4, 5, 6]), counts=ivy.array([1, 2, 3, 3, 2, 1]))

    >>> x = ivy.array([0.2,0.3,0.4,0.2,1.4,2.3,0.2])
    >>> y = ivy.unique_counts(x)
    >>> print(y)
    Results(values=ivy.array([0.2       , 0.30000001, 0.40000001, 1.39999998,
                              2.29999995]),
            counts=ivy.array([3, 1, 1, 1, 1]))

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 3. , 2. , 1. , 0.]),
    ...                   b=ivy.array([1, 2, 1, 3, 4, 1, 3]))
    >>> y = ivy.unique_counts(x)
    >>> print(y)
    {
        a: (list[2],<classivy.array.array.Array>shape=[4]),
        b: (list[2],<classivy.array.array.Array>shape=[4])
    }
    """
    return ivy.current_backend(x).unique_counts(x)
