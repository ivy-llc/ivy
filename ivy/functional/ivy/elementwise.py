# global
from numbers import Number
from typing import Optional, Union

# local
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
    handle_device_shifting,
)
from ivy.utils.exceptions import handle_exceptions


# Array API Standard #
# -------------------#


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def abs(
    x: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:  # noqa
    """Calculate the absolute value for each element ``x_i`` of the input array ``x``
    (i.e., the element-wise result has the same magnitude as the respective element in
    ``x`` but has positive sign).

    .. note::
        For signed integer data types, the absolute value of the minimum representable
        integer is implementation-dependent.

    **Special Cases**

    For real-valued floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``-0``, the result is ``+0``.
    - If ``x_i`` is ``-infinity``, the result is ``+infinity``.

    For complex floating-point operands,
    let ``a = real(x_i)`` and ``b = imag(x_i)``. and

    - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is any value
      (including ``NaN``), the result is ``+infinity``.
    - If ``a`` is any value (including ``NaN``) and ``b`` is ``+infinity``,
      the result is ``+infinity``.
    - If ``a`` is either ``+0`` or ``-0``, the result is ``abs(b)``.
    - If ``b`` is ``+0`` or ``-0``, the result is ``abs(a)``.
    - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN``.
    - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN``.
    - If ``a`` is ``Na``N and ``b`` is ``NaN``, the result is ``NaN``.

    Parameters
    ----------
    x
        input array. Should have a numeric data type

    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.


    Returns
    -------
    ret
        an array containing the absolute value of each element in ``x``. The returned
        array must have the same data type as ``x``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.abs.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1,0,-6])
    >>> y = ivy.abs(x)
    >>> print(y)
    ivy.array([1, 0, 6])

    >>> x = ivy.array([3.7, -7.7, 0, -2, -0])
    >>> y = ivy.abs(x)
    >>> print(y)
    ivy.array([ 3.7, 7.7, 0., 2., 0.])

    >>> x = ivy.array([[1.1, 2.2, 3.3], [-4.4, -5.5, -6.6]])
    >>> ivy.abs(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1,  2.2,  3.3],
               [4.4, 5.5, 6.6]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]), b=ivy.array([4.5, -5.3, -0, -2.3])) # noqa
    >>> y = ivy.abs(x)
    >>> print(y)
    {
        a: ivy.array([0., 2.6, 3.5]),
        b: ivy.array([4.5, 5.3, 0., 2.3])
    }

    """

    return ivy.current_backend(x).abs(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def acos(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate an implementation-dependent approximation of the principal value of the
    inverse cosine, having domain [-1, +1] and codomain [+0, +π], for each element x_i
    of the input array x. Each element-wise result is expressed in radians.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is greater than ``1``, the result is ``NaN``.
    - If ``x_i`` is less than ``-1``, the result is ``NaN``.
    - If ``x_i`` is ``1``, the result is ``+0``.

    For complex floating-point operands, let a = real(x_i) and b = imag(x_i),
    and

    - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``,
      the result is ``π/2 - 0j``.
    - if ``a`` is either ``+0`` or ``-0`` and ``b`` is ``NaN``,
      the result is ``π/2 + NaN j``.
    - If ``a`` is a finite number and ``b`` is ``+infinity``,
      the result is ``π/2 - infinity j``.
    - If ``a`` is a nonzero finite number and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``-infinity`` and ``b`` is a positive
      (i.e., greater than 0) finite number, the result is ``π - infinity j``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive
      (i.e., greater than 0) finite number, the result is ``+0 - infinity j``.
    - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``,
      the result is ``3π/4 - infinity j``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``π/4 - infinity j``.
    - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is ``NaN``,
      the result is ``NaN ± infinity j`` (sign of
      the imaginary component is unspecified).
    - If ``a`` is ``NaN`` and ``b`` is a finite number,
      the result is ``NaN + NaN j``.
    - if ``a`` is ``NaN`` and ``b`` is ``+infinity``,
      the result is ``NaN - infinity j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the inverse cosine of each element in x. The returned array
        must have a floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.acos.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments


    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., -1.])
    >>> y = ivy.acos(x)
    >>> print(y)
    ivy.array([1.57, 0.  , 3.14])

    >>> x = ivy.array([1., 0., -1.])
    >>> y = ivy.zeros(3)
    >>> ivy.acos(x, out=y)
    >>> print(y)
    ivy.array([0.  , 1.57, 3.14])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -1]))
    >>> y = ivy.acos(x)
    >>> print(y)
    {
        a: ivy.array([1.57, 3.14, 0.]),
        b: ivy.array([0., 1.57, 3.14])
    }
    """
    return ivy.current_backend(x).acos(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def acosh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation to the inverse hyperbolic
    cosine, having domain ``[+1, +infinity]`` and codomain ``[+0, +infinity]``, for each
    element ``x_i`` of the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``1``, the result is ``NaN``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    For complex floating-point operands, let a = real(x_i) and b = imag(x_i),
    and

    - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``,
      the result is ``+0 + πj/2``.
    - If ``a`` is a finite number and ``b`` is ``+infinity``,
      the result is ``+infinity + πj/2``.
    - If ``a`` is a nonzero finite number and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``+0`` and ``b`` is ``NaN``, the result is ``NaN ± πj/2``
      (sign of the imaginary component is unspecified).
    - If ``a`` is ``-infinity`` and ``b`` is a positive (i.e., greater than 0) finite
      number, the result is ``+infinity + πj``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than 0) finite
      number, the result is ``+infinity + 0j``.
    - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``,
      the result is ``+infinity + 3πj/4``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``+infinity + πj/4``.
    - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is ``NaN``,
      the result is ``+infinity + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is a finite number,
      the result is ``NaN + NaN j``.
    - if ``a`` is ``NaN`` and ``b`` is ``+infinity``,
      the result is ``+infinity + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

    Parameters
    ----------
    x
        input array whose elements each represent the area of a hyperbolic sector.
        Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the inverse hyperbolic cosine of each element in x. The
        returned array must have a floating-point data type determined by
        :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.acosh.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2.5, 10])
    >>> y = ivy.acosh(x)
    >>> print(y)
    ivy.array([0.  , 1.57, 2.99])

    >>> x = ivy.array([1., 2., 6.])
    >>> y = ivy.zeros(3)
    >>> ivy.acosh(x, out=y)
    >>> print(y)
    ivy.array([0.  , 1.32, 2.48])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2, 10]), b=ivy.array([1., 10, 6]))
    >>> y = ivy.acosh(x)
    >>> print(y)
    {
        a: ivy.array([0., 1.32, 2.99]),
        b: ivy.array([0., 2.99, 2.48])
    }


    """
    return ivy.current_backend(x).acosh(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def add(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate the sum for each element ``x1_i`` of the input array ``x1`` with the
    respective element ``x2_i`` of the input array ``x2``.

    **Special cases**

    For floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``-infinity``, the result is ``NaN``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``+infinity``, the result is ``NaN``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``+infinity``, the result is
      ``+infinity``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``-infinity``, the result is
      ``-infinity``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a finite number, the result is
      ``+infinity``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a finite number, the result is
      ``-infinity``.
    - If ``x1_i`` is a finite number and ``x2_i`` is ``+infinity``, the result is
      ``+infinity``.
    - If ``x1_i`` is a finite number and ``x2_i`` is ``-infinity``, the result is
      ``-infinity``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is ``-0``, the result is ``-0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is ``+0``, the result is ``+0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is ``-0``, the result is ``+0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is ``+0``, the result is ``+0``.
    - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is a nonzero finite number,
      the result is ``x2_i``.
    - If ``x1_i`` is a nonzero finite number and ``x2_i`` is either ``+0`` or ``-0``,
      the result is ``x1_i``.
    - If ``x1_i`` is a nonzero finite number and ``x2_i`` is ``-x1_i``, the result is
      ``+0``.
    - In the remaining cases, when neither ``infinity``, ``+0``, ``-0``, nor a ``NaN``
      is involved, and the operands have the same mathematical sign or have different
      magnitudes, the sum must be computed and rounded to the nearest representable
      value according to IEEE 754-2019 and a supported round mode. If the magnitude is
      too large to represent, the operation overflows and the result is an `infinity`
      of appropriate mathematical sign.

    .. note::
       Floating-point addition is a commutative operation, but not always associative.

    For complex floating-point operands, addition is defined according
    to the following table.
    For real components ``a`` and ``c``, and imaginary components ``b`` and ``d``,

    +-------------------+-------------------+-------------------+-------------------+
    |                   |         c         |       dj          |         c+dj      |
    +===================+===================+===================+===================+
    |     **a**         |       a + c       |      a + dj       |  (a+c) + dj       |
    +-------------------+-------------------+-------------------+-------------------+
    |     **bj**        |    c + bj         |     (b+d)j        |   c + (b+d)j      |
    +-------------------+-------------------+-------------------+-------------------+
    |     **a+bj**      | (a+c) + bj        |   a + (b+d)j      | (a+c) + (b+d)j    |
    +-------------------+-------------------+-------------------+-------------------+

    For complex floating-point operands, the real valued floating-point
    special cases must independently apply to the real and
    imaginary component operation invloving real numbers as
    described in the above table. For example, let ``a = real(x1_i)``,
    ``c = real(x2_i)``, ``d = imag(x2_i)``,
    and
    - if ``a`` is ``-0``, the real component of the result is ``-0``.
    - Similarly, if ``b`` is ``+0`` and ``d`` is ``-0``,
    the imaginary component of the result is ``+0``.

    Hence, if ``z1 = a + bj = -0 + 0j`` and ``z2 = c + dj = -0 - 0j``,
    then the result of  ``z1 + z2`` is ``-0 + 0j``.

    Parameters
    ----------
    x1
        first input array. Should have a numeric data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have a numeric data type.
    alpha
        optional scalar multiplier for ``x2``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise sums. The returned array must have a data
        type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.add.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = ivy.add(x, y)
    >>> print(z)
    ivy.array([5, 7, 9])

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = ivy.add(x, y, alpha=2)
    >>> print(z)
    ivy.array([9, 12, 15])

    >>> x = ivy.array([[1.1, 2.3, -3.6]])
    >>> y = ivy.array([[4.8], [5.2], [6.1]])
    >>> z = ivy.zeros((3, 3))
    >>> ivy.add(x, y, out=z)
    >>> print(z)
    ivy.array([[5.9, 7.1, 1.2],
               [6.3, 7.5, 1.6],
               [7.2, 8.4, 2.5]])

    >>> x = ivy.array([[[1.1], [3.2], [-6.3]]])
    >>> y = ivy.array([[8.4], [2.5], [1.6]])
    >>> ivy.add(x, y, out=x)
    >>> print(x)
    ivy.array([[[9.5],
                [5.7],
                [-4.7]]])
    """
    return ivy.current_backend(x1, x2).add(x1, x2, alpha=alpha, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def asin(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation of the principal value of
    the inverse sine, having domain ``[-1, +1]`` and codomain ``[-π/2, +π/2]`` for each
    element ``x_i`` of the input array ``x``. Each element-wise result is expressed in
    radians.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is greater than ``1``, the result is ``NaN``.
    - If ``x_i`` is less than ``-1``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.

    For complex floating-point operands, special cases must be handled
    as if the operation is implemented as ``-1j * asinh(x * 1j)``.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the inverse sine of each element in ``x``. The returned
        array must have a floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.asin.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([-2.4, -0, +0, 3.2, float('nan')])
    >>> y = ivy.asin(x)
    >>> print(y)
    ivy.array([nan,  0.,  0., nan, nan])

    >>> x = ivy.array([-1, -0.5, 0.6, 1])
    >>> y = ivy.zeros(4)
    >>> ivy.asin(x, out=y)
    >>> print(y)
    ivy.array([-1.57,-0.524,0.644,1.57])

    >>> x = ivy.array([[0.1, 0.2, 0.3],[-0.4, -0.5, -0.6]])
    >>> ivy.asin(x, out=x)
    >>> print(x)
    ivy.array([[0.1,0.201,0.305],[-0.412,-0.524,-0.644]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 0.1, 0.2]),
    ...                   b=ivy.array([0.3, 0.4, 0.5]))
    >>> y = ivy.asin(x)
    >>> print(y)
    {a:ivy.array([0.,0.1,0.201]),b:ivy.array([0.305,0.412,0.524])}
    """
    return ivy.current_backend(x).asin(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def asinh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation to the inverse hyperbolic
    sine, having domain ``[-infinity, +infinity]`` and codomain
    ``[-infinity, +infinity]``, for each element ``x_i`` in the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.

    For complex floating-point operands, let ``a = real(x_i)`` and ``b = imag(x_i)``,
    and

    - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
    - If ``a`` is a positive (i.e., greater than ``0``) finite number and ``b`` is
      ``+infinity``, the result is ``+infinity + πj/2``.
    - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``)
      finite number, the result is ``+infinity + 0j``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is
      ``+infinity + πj/4``.
    - If ``a`` is ``NaN`` and ``b`` is ``+0``, the result is ``NaN + 0j``.
    - If ``a`` is ``NaN`` and ``b`` is nonzero finite number, the result is
      ``NaN + NaNj``.
    - If ``a`` is ``NaN`` and ``b`` is ``+infinity``,
      the result is ``±infinity ± NaNj``, (sign of real component is unspecified).
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``, ``NaN + NaNj``.

    Parameters
    ----------
    x
        input array whose elements each represent the area of a hyperbolic sector.
        Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the inverse hyperbolic sine of each element in ``x``. The
        returned array must have a floating-point data type determined by
        :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.asinh.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-3.5, -0, +0, 1.3, float('nan')])
    >>> y = ivy.asinh(x)
    >>> print(y)
    ivy.array([-1.97, 0., 0., 1.08, nan])

    >>> x = ivy.array([-2, -0.75, 0.9, 1])
    >>> y = ivy.zeros(4)
    >>> ivy.asinh(x, out=y)
    >>> print(y)
    ivy.array([-1.44, -0.693, 0.809, 0.881])

    >>> x = ivy.array([[0.2, 0.4, 0.6],[-0.8, -1, -2]])
    >>> ivy.asinh(x, out=x)
    >>> print(x)
    ivy.array([[ 0.199, 0.39, 0.569],
               [-0.733, -0.881, -1.44]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1, 2]),
    ...                   b=ivy.array([4.2, -5.3, -0, -2.3]))
    >>> y = ivy.asinh(x)
    >>> print(y)
    {
        a: ivy.array([0., 0.881, 1.44]),
        b: ivy.array([2.14, -2.37, 0., -1.57])
    }

    """
    return ivy.current_backend(x).asinh(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def atan(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation of the principal value of
    the inverse tangent, having domain ``[-infinity, +infinity]`` and codomain
    ``[-π/2, +π/2]``, for each element ``x_i`` of the input array ``x``. Each
    element-wise result is expressed in radians.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is an implementation-dependent
      approximation to ``+π/2``.
    - If ``x_i`` is ``-infinity``, the result is an implementation-dependent
      approximation to ``-π/2``.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the inverse tangent of each element in ``x``. The returned
        array must have a floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.atan.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.atan(x)
    >>> print(y)
    ivy.array([0.   , 0.785, 1.11 ])

    >>> x = ivy.array([4., 0., -6.])
    >>> y = ivy.zeros(3)
    >>> ivy.atan(x, out=y)
    >>> print(y)
    ivy.array([ 1.33,  0.  , -1.41])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -6]))
    >>> y = ivy.atan(x)
    >>> print(y)
    {
        a: ivy.array([0., -0.785, 0.785]),
        b: ivy.array([0.785, 0., -1.41])
    }

    """
    return ivy.current_backend(x).atan(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def atan2(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation of the inverse tangent of
    the quotient ``x1/x2``, having domain ``[-infinity, +infinity] x.
    [-infinity, +infinity]`` (where the ``x`` notation denotes the set of ordered pairs
    of elements ``(x1_i, x2_i)``) and codomain ``[-π, +π]``, for each pair of elements
    ``(x1_i, x2_i)`` of the input arrays ``x1`` and ``x2``, respectively. Each
    element-wise result is expressed in radians. The mathematical signs of
    ``x1_i and x2_i`` determine the quadrant of each element-wise result. The quadrant
    (i.e., branch) is chosen such that each element-wise result is the signed angle in
    radians between the ray ending at the origin and passing through the point ``(1,0)``
    and the ray ending at the origin and passing through the point ``(x2_i, x1_i)``.

    **Special cases**

    For floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is an
      approximation to ``+π/2``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is an
      approximation to ``+π/2``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is ``+0``, the result is ``+0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is ``-0``, the result is an approximation to
      ``+π``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is less than 0, the result is an approximation
      to ``+π``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``-0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is ``+0``, the result is ``-0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is ``-0``, the result is an approximation to
      ``-π``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is an
      approximation to ``-π``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is an
      approximation to ``-π/2``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is an
      approximation to ``-π/2``.
    - If ``x1_i`` is greater than ``0``, ``x1_i`` is a finite number, and ``x2_i`` is
      ``+infinity``, the result is ``+0``.
    - If ``x1_i`` is greater than ``0``, ``x1_i`` is a finite number, and ``x2_i`` is
      ``-infinity``, the result is an approximation to ``+π``.
    - If ``x1_i`` is less than ``0``, ``x1_i`` is a finite number, and ``x2_i`` is
      ``+infinity``, the result is ``-0``.
    - If ``x1_i`` is less than ``0``, ``x1_i`` is a finite number, and ``x2_i`` is
      ``-infinity``, the result is an approximation to ``-π``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is finite, the result is an
      approximation to ``+π/2``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is finite, the result is an
      approximation to ``-π/2``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``+infinity``, the result is an
      approximation to ``+π/4``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``-infinity``, the result is an
      approximation to ``+3π/4``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``+infinity``, the result is an
      approximation to ``-π/4``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``-infinity``, the result is an
      approximation to ``-3π/4``.

    Parameters
    ----------
    x1
        input array corresponding to the y-coordinates. Should have a floating-point
        data type.
    x2
        input array corresponding to the x-coordinates. Must be compatible with ``x1``.
        Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the inverse tangent of the quotient ``x1/x2``. The returned
        array must have a floating-point data type.


    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.atan2.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1.0, -1.0, -2.0])
    >>> y = ivy.array([2.0, 0.0, 3.0])
    >>> z = ivy.atan2(x, y)
    >>> print(z)
    ivy.array([ 0.464, -1.57 , -0.588])

    >>> x = ivy.array([1.0, 2.0])
    >>> y = ivy.array([-2.0, 3.0])
    >>> z = ivy.zeros(2)
    >>> ivy.atan2(x, y, out=z)
    >>> print(z)
    ivy.array([2.68 , 0.588])

    >>> nan = float("nan")
    >>> x = ivy.array([nan, 1.0, 1.0, -1.0, -1.0])
    >>> y = ivy.array([1.0, +0, -0, +0, -0])
    >>> z = ivy.atan2(x, y)
    >>> print(z)
    ivy.array([  nan,  1.57,  1.57, -1.57, -1.57])

    >>> x = ivy.array([+0, +0, +0, +0, -0, -0, -0, -0])
    >>> y = ivy.array([1.0, +0, -0, -1.0, 1.0, +0, -0, -1.0])
    >>> z = ivy.atan2(x, y)
    >>> print(z)
    ivy.array([0.  , 0.  , 0.  , 3.14, 0.  , 0.  , 0.  , 3.14])

    >>> inf = float("infinity")
    >>> x = ivy.array([inf, -inf, inf, inf, -inf, -inf])
    >>> y = ivy.array([1.0, 1.0, inf, -inf, inf, -inf])
    >>> z = ivy.atan2(x, y)
    >>> print(z)
    ivy.array([ 1.57 , -1.57 ,  0.785,  2.36 , -0.785, -2.36 ])

    >>> x = ivy.array([2.5, -1.75, 3.2, 0, -1.0])
    >>> y = ivy.array([-3.5, 2, 0, 0, 5])
    >>> z = ivy.atan2(x, y)
    >>> print(z)
    ivy.array([ 2.52 , -0.719,  1.57 ,  0.   , -0.197])

    >>> x = ivy.array([[1.1, 2.2, 3.3], [-4.4, -5.5, -6.6]])
    >>> y = ivy.atan2(x, x)
    >>> print(y)
    ivy.array([[ 0.785,  0.785,  0.785],
        [-2.36 , -2.36 , -2.36 ]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),
    ...                   b=ivy.array([4.5, -5.3, -0]))
    >>> y = ivy.array([3.0, 2.0, 1.0])
    >>> z = ivy.atan2(x, y)
    {
        a: ivy.array([0., 0.915, -1.29]),
        b: ivy.array([0.983, -1.21, 0.])
    }

    >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),
    ...                   b=ivy.array([4.5, -5.3, -0, -2.3]))
    >>> y = ivy.Container(a=ivy.array([-2.5, 1.75, 3.5]),
    ...                   b=ivy.array([2.45, 6.35, 0, 1.5]))
    >>> z = ivy.atan2(x, y)
    >>> print(z)
    {
        a: ivy.array([3.14, 0.978, -0.785]),
        b: ivy.array([1.07, -0.696, 0., -0.993])
    }
    """
    return ivy.current_backend(x1).atan2(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def atanh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array with the inverse hyperbolic tangent of the elements of ``x``.

    Parameters
    ----------
    x
        input array whose elements each represent the area of a hyperbolic sector.
        Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the inverse hyperbolic tangent of each element in ``x``. The
        returned array must have a floating-point data type determined by Type Promotion
        Rules.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.atanh.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    **Special cases**

    For real-valued floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``-1``, the result is ``NaN``.
    - If ``x_i`` is greater than ``1``, the result is ``NaN``.
    - If ``x_i`` is ``-1``, the result is ``-infinity``.
    - If ``x_i`` is ``+1``, the result is ``+infinity``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.

    For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

    - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
    - If ``a`` is ``+0`` and ``b`` is ``NaN``, the result is ``+0 + NaN j``.
    - If ``a`` is ``1`` and ``b`` is ``+0``, the result is ``+infinity + 0j``.
    - If ``a`` is a positive (i.e., greater than ``0``) finite number and ``b``
      is ``+infinity``, the result is ``+0 + πj/2``.
    - If ``a`` is a nonzero finite number and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``)
      finite number, the result is ``+0 + πj/2``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``+0 + πj/2``.
    - If ``a`` is ``+infinity`` and ``b`` is ``NaN``, the result is ``+0 + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``+infinity``, the result is ``±0 + πj/2``
      (sign of the real component is unspecified).
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, -0.5])
    >>> y = ivy.atanh(x)
    >>> print(y)
    ivy.array([ 0.   , -0.549])

    >>> x = ivy.array([0.5, -0.5, 0.])
    >>> y = ivy.zeros(3)
    >>> ivy.atanh(x, out=y)
    >>> print(y)
    ivy.array([ 0.549, -0.549,  0.   ])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., -0.5]), b=ivy.array([ 0., 0.5]))
    >>> y = ivy.atanh(x)
    >>> print(y)
    {
        a: ivy.array([0., -0.549]),
        b: ivy.array([0., 0.549])
    }


    """
    return ivy.current_backend(x).atanh(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def bitwise_and(
    x1: Union[int, bool, ivy.Array, ivy.NativeArray],
    x2: Union[int, bool, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the bitwise AND of the underlying binary representation of each element
    ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have an integer or boolean data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have an integer or boolean data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.bitwise_and.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Functional Examples
    -------------------

    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([2, 3, 7])
    >>> y = ivy.array([7, 1, 15])
    >>> z = ivy.bitwise_and(x, y)
    >>> print(z)
    ivy.array([2, 1, 7])

    >>> x = ivy.array([[True], [False]])
    >>> y = ivy.array([[True], [True]])
    >>> ivy.bitwise_and(x, y, out=x)
    >>> print(x)
    ivy.array([[ True],[False]])

    >>> x = ivy.array([1])
    >>> y = ivy.array([3])
    >>> ivy.bitwise_and(x, y, out=y)
    >>> print(y)
    ivy.array([1])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
    >>> y = ivy.Container(a=ivy.array([7, 8, 9]), b=ivy.array([10, 11, 11]))
    >>> z = ivy.bitwise_and(x, y)
    >>> print(z)
    {
        a: ivy.array([1, 0, 1]),
        b: ivy.array([0, 1, 2])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([True, True])
    >>> y = ivy.Container(a=ivy.array([True, False]), b=ivy.array([False, True]))
    >>> z = ivy.bitwise_and(x, y)
    >>> print(z)
    {
        a: ivy.array([True, False]),
        b: ivy.array([False, True])
    }
    """
    return ivy.current_backend(x1, x2).bitwise_and(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def bitwise_invert(
    x: Union[int, bool, ivy.Array, ivy.NativeArray, ivy.Container],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Inverts (flips) each bit for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have an integer or boolean data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have the
        same data type as x.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.bitwise_invert.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 6, 9])
    >>> y = ivy.bitwise_invert(x)
    >>> print(y)
    ivy.array([-2, -7, -10])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=[False, True, False], b=[True, True, False])
    >>> y = ivy.bitwise_invert(x)
    >>> print(y)
    {
        a: ivy.array([True, False, True]),
        b: ivy.array([False, False, True])
    }

    With :class:`int` input:

    >>> x = -8
    >>> y = ivy.bitwise_invert(x)
    >>> print(y)
    ivy.array(7)

    With :class:`bool` input:

    >>> x = False
    >>> y = ivy.bitwise_invert(x)
    >>> print(y)
    ivy.array(True)
    """
    return ivy.current_backend(x).bitwise_invert(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def bitwise_left_shift(
    x1: Union[int, ivy.Array, ivy.NativeArray],
    x2: Union[int, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the left by
    appending ``x2_i`` (i.e., the respective element in the input array ``x2``) zeros to
    the right of ``x1_i``.

    Parameters
    ----------
    x1
        first input array. Should have an integer data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have an integer data type. Each element must be greater than or equal to
        ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.bitwise_left_shift.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    """
    return ivy.current_backend(x1, x2).bitwise_left_shift(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def bitwise_or(
    x1: Union[int, bool, ivy.Array, ivy.NativeArray],
    x2: Union[int, bool, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the bitwise OR of the underlying binary representation of each element
    ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have an integer or boolean data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have an integer or boolean data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type determined by :ref:`type-promotion`.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = ivy.bitwise_or(x, y)
    >>> print(z)
    ivy.array([5, 7, 7])

    >>> x = ivy.array([[[1], [2], [3], [4]]])
    >>> y = ivy.array([[[4], [5], [6], [7]]])
    >>> ivy.bitwise_or(x, y, out=x)
    >>> print(x)
    ivy.array([[[5],
                [7],
                [7],
                [7]]])

    >>> x = ivy.array([[[1], [2], [3], [4]]])
    >>> y = ivy.array([4, 5, 6, 7])
    >>> z = ivy.bitwise_or(x, y)
    >>> print(z)
    ivy.array([[[5, 5, 7, 7],
                [6, 7, 6, 7],
                [7, 7, 7, 7],
                [4, 5, 6, 7]]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([5, 6, 7]))
    >>> z = ivy.bitwise_or(x, y)
    >>> print(z)
    {
        a: ivy.array([5, 7, 7]),
        b: ivy.array([7, 7, 7])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([5, 6, 7]))
    >>> z = ivy.bitwise_or(x, y)
    >>> print(z)
    {
        a: ivy.array([5,7,7]),
        b: ivy.array([5,6,7])
    }

    """
    return ivy.current_backend(x1, x2).bitwise_or(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def bitwise_right_shift(
    x1: Union[int, ivy.Array, ivy.NativeArray],
    x2: Union[int, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the
    right according to the respective element ``x2_i`` of the input array ``x2``.

    .. note::
       This operation must be an arithmetic shift (i.e., sign-propagating) and thus
       equivalent to floor division by a power of two.

    Parameters
    ----------
    x1
        first input array. Should have an integer data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have an integer data type. Each element must be greater than or equal
        to ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.bitwise_right_shift.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> a = ivy.array([2, 9, 16, 31])
    >>> b = ivy.array([0, 1, 2, 3])
    >>> y = ivy.bitwise_right_shift(a, b)
    >>> print(y)
    ivy.array([2, 4, 4, 3])

    >>> a = ivy.array([[32, 40, 55], [16, 33, 170]])
    >>> b = ivy.array([5, 2, 1])
    >>> y = ivy.zeros((2, 3))
    >>> ivy.bitwise_right_shift(a, b, out=y)
    >>> print(y)
    ivy.array([[ 1., 10., 27.],
               [ 0.,  8., 85.]])

    >>> a = ivy.array([[10, 64],[43, 87],[5, 37]])
    >>> b = ivy.array([1, 3])
    >>> ivy.bitwise_right_shift(a, b, out=a)
    >>> print(a)
    ivy.array([[ 5,  8],
               [21, 10],
               [ 2,  4]])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> a = ivy.array([[10, 64],[43, 87],[5, 37]])
    >>> b = ivy.native_array([1, 3])
    >>> y = ivy.bitwise_right_shift(a, b)
    >>> print(y)
    ivy.array([[ 5,  8],[21, 10],[ 2,  4]])

    With one :class:`ivy.Container` input:

    >>> a = ivy.Container(a = ivy.array([100, 200]),
    ...                   b = ivy.array([125, 243]))
    >>> b = ivy.array([3, 6])
    >>> y = ivy.bitwise_right_shift(a, b)
    >>> print(y)
    {
        a: ivy.array([12, 3]),
        b: ivy.array([15, 3])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> a = ivy.Container(a = ivy.array([10, 25, 42]),
    ...                   b = ivy.array([64, 65]),
    ...                   c = ivy.array([200, 225, 255]))
    >>> b = ivy.Container(a = ivy.array([0, 1, 2]),
    ...                   b = ivy.array([6]),
    ...                   c = ivy.array([4, 5, 6]))
    >>> y = ivy.bitwise_right_shift(a, b)
    >>> print(y)
    {
        a: ivy.array([10, 12, 10]),
        b: ivy.array([1, 1]),
        c: ivy.array([12, 7, 3])
    }
    """
    return ivy.current_backend(x1, x2).bitwise_right_shift(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def bitwise_xor(
    x1: Union[int, bool, ivy.Array, ivy.NativeArray],
    x2: Union[int, bool, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the bitwise XOR of the underlying binary representation of each element
    ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the
    input
    array ``x2``.

    **Special cases**

    This function does not take floating point operands

    Parameters
    ----------
    x1
        first input array. Should have an integer or boolean data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have an integer or boolean data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.bitwise_xor.html>`_
    in the standard.

    Both the description and the type hints above assume an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> a = ivy.array([1, 2, 3])
    >>> b = ivy.array([3, 2, 1])
    >>> y = ivy.bitwise_xor(a, b)
    >>> print(y)
    ivy.array([2, 0, 2])

    >>> a = ivy.array([78, 91, 23])
    >>> b = ivy.array([66, 77, 88])
    >>> ivy.bitwise_xor(a, b, out=y)
    >>> print(y)
    ivy.array([12, 22, 79])

    >>> a = ivy.array([1, 2, 3])
    >>> b = ivy.array([3, 2, 1])
    >>> ivy.bitwise_xor(a, b, out = a)
    >>> print(a)
    ivy.array([2, 0, 2])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> a = ivy.array([0, 1, 3, 67, 91])
    >>> a = ivy.native_array([4, 7, 90, 89, 98])
    >>> y = ivy.bitwise_xor(a, b)
    >>> print(y)
    ivy.array([0,0,0,0,0])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a = ivy.array([89]))
    >>> y = ivy.Container(a = ivy.array([12]))
    >>> z = ivy.bitwise_xor(x, y)
    >>> print(z)
    {
        a:ivy.array([85])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a = ivy.array([-67, 21]))
    >>> b = ivy.array([78, 34])
    >>> y = ivy.array([12, 13])
    >>> z = ivy.bitwise_xor(x, y)
    >>> print(z)
    {
        a: ivy.array([-79, 24])
    }
    """
    return ivy.current_backend(x1, x2).bitwise_xor(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def ceil(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Round each element ``x_i`` of the input array ``x`` to the smallest (i.e.,
    closest to ``-infinity``) integer-valued number that is not less than ``x_i``.

    **Special cases**

    - If ``x_i`` is already integer-valued, the result is ``x_i``.

    For floating-point operands,

    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``NaN``, the result is ``NaN``.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rounded result for each element in ``x``. The returned
        array must have the same data type as ``x``.


    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.ceil.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.1, 0, -0.1])
    >>> y = ivy.ceil(x)
    >>> print(y)
    ivy.array([1., 0., -0.])

    >>> x = ivy.array([2.5, -3.5, 0, -3, -0])
    >>> y = ivy.ones(5)
    >>> ivy.ceil(x, out=y)
    >>> print(y)
    ivy.array([ 3., -3.,  0., -3.,  0.])

    >>> x = ivy.array([[3.3, 4.4, 5.5], [-6.6, -7.7, -8.8]])
    >>> ivy.ceil(x, out=x)
    >>> print(x)
    ivy.array([[ 4.,  5.,  6.],
               [-6., -7., -8.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([2.5, 0.5, -1.4]),
    ...                   b=ivy.array([5.4, -3.2, -0, 5.2]))
    >>> y = ivy.ceil(x)
    >>> print(y)
    {
        a: ivy.array([3., 1., -1.]),
        b: ivy.array([6., -3., 0., 6.])
    }
    """
    return ivy.current_backend(x).ceil(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def cos(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation to the cosine, having domain
    ``(-infinity, +infinity)`` and codomain ``[-1, +1]``, for each element ``x_i`` of
    the input array ``x``. Each element ``x_i`` is assumed to be expressed in radians.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``1``.
    - If ``x_i`` is ``-0``, the result is ``1``.
    - If ``x_i`` is ``+infinity``, the result is ``NaN``.
    - If ``x_i`` is ``-infinity``, the result is ``NaN``.

    For complex floating-point operands, special cases must be handled as if the
    operation is implemented as ``cosh(x*1j)``.

    Parameters
    ----------
    x
        input array whose elements are each expressed in radians. Should have a
        floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the cosine of each element in ``x``. The returned array must
        have a floating-point data type determined by :ref:`type-promotion`.


    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.cos.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.cos(x)
    >>> print(y)
    ivy.array([1., 0.54, -0.416])

    >>> x = ivy.array([4., 0., -6.])
    >>> y = ivy.zeros(3)
    >>> ivy.cos(x, out=y)
    >>> print(y)
    ivy.array([-0.654, 1., 0.96])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -6]))
    >>> y = ivy.cos(x)
    >>> print(y)
    {
        a: ivy.array([1., 0.54, 0.54]),
        b: ivy.array([0.54, 1., 0.96])
    }
    """
    return ivy.current_backend(x).cos(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def cosh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation to the hyperbolic cosine,
    having domain ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``,
    for each element ``x_i`` in the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``1``.
    - If ``x_i`` is ``-0``, the result is ``1``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``+infinity``.

    For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

    .. note::
       For complex floating-point operands, ``cosh(conj(x))``
       must equal ``conj(cosh(x))``.

    - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``1 + 0j``.
    - If ``a`` is ``+0`` and ``b`` is ``+infinity``, the result is ``NaN + 0j``
      (sign of the imaginary component is unspecified).
    - If ``a`` is ``+0`` and ``b`` is ``NaN``, the result is ``NaN + 0j``
      (sign of the imaginary component is unspecified).
    - If ``a`` is a nonzero finite number and ``b`` is ``+infinity``,
      the result is ``NaN + NaN j``.
    - If ``a`` is a nonzero finite number and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+0``,
      the result is ``+infinity + 0j``.
    - If ``a`` is ``+infinity`` and ``b`` is a nonzero finite number,
      the result is ``+infinity * cis(b)``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``+infinity + NaN j``(sign of the real component is unspecified).
    - If ``a`` is ``+infinity`` and ``b`` is ``NaN``,
      the result is ``+infinity + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is either ``+0`` or ``-0``,
      the result is ``NaN + 0j`` (sign of the imaginary component is unspecified).
    - If ``a`` is ``NaN`` and ``b`` is a nonzero finite number,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

    where ``cis(v)`` is ``cos(v) + sin(v)*1j``.

    Parameters
    ----------
    x
        input array whose elements each represent a hyperbolic angle. Should have a
        floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the hyperbolic cosine of each element in ``x``. The returned
        array must have a floating-point data type determined by :ref:`type-promotion`.


    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.cosh.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3, 4])
    >>> y = ivy.cosh(x)
    >>> print(y)
    ivy.array([1.54,3.76,10.1,27.3])

    >>> x = ivy.array([0.2, -1.7, -5.4, 1.1])
    >>> y = ivy.zeros(4)
    >>> ivy.cosh(x, out=y)
    ivy.array([[1.67,4.57,13.6,12.3],[40.7,122.,368.,670.]])

    >>> x = ivy.array([[1.1, 2.2, 3.3, 3.2],
    ...                [-4.4, -5.5, -6.6, -7.2]])
    >>> y = ivy.cosh(x)
    >>> print(y)
    ivy.array([[1.67,4.57,13.6,12.3],[40.7,122.,368.,670.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1., 2., 3.]), b=ivy.array([6., 7., 8.]))
    >>> y = ivy.cosh(x)
    >>> print(y)
    {
        a:ivy.array([1.54,3.76,10.1]),
        b:ivy.array([202.,548.,1490.])
    }
    """
    return ivy.current_backend(x).cosh(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def divide(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""
    Calculate the division for each element x1_i of the input array x1 with the
    respective element x2_i of the input array x2.

    **Special Cases**

    For real-valued floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i``
      is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
    - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i``
      is either ``+0`` or ``-0``, the result is ``NaN``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``,
      the result is ``+0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``,
      the result is ``-0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``,
      the result is ``-0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``,
      the result is ``+0``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``,
      the result is ``+infinity``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``,
      the result is ``-infinity``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``,
      the result is ``-infinity``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``,
      the result is ``+infinity``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``+infinity``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative
      (i.e., less than ``0``) finite number, the result is ``-infinity``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``-infinity``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative
      (i.e., less than ``0``) finite number, the result is ``+infinity``.
    - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and
      ``x2_i`` is ``+infinity``, the result is ``+0``.
    - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and
      ``x2_i`` is ``-infinity``, the result is ``-0``.
    - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and
      ``x2_i`` is ``+infinity``, the result is ``-0``.
    - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and
      ``x2_i`` is ``-infinity``, the result is ``+0``.
    - If ``x1_i`` and ``x2_i`` have the same mathematical sign and
      are both nonzero finite numbers, the result has a positive mathematical sign.
    - If ``x1_i`` and ``x2_i`` have different mathematical signs and
      are both nonzero finite numbers, the result has a negative mathematical sign.
    - In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN``
      is involved, the quotient must be computed and rounded
      to the nearest representable value according to IEEE 754-2019
      and a supported rounding mode.
      If the magnitude is too large to represent, the operation overflows and
      the result is an ``infinity`` of appropriate mathematical sign.
      If the magnitude is too small to represent, the operation
      underflows and the result is a zero of appropriate mathematical sign.

    For complex floating-point operands,
    division is defined according to the following table.
    For real components ``a`` and ``c`` and imaginary components ``b`` and ``d``,

    +------------+----------------+-----------------+--------------------------+
    |            | c              | dj              | c + dj                   |
    +============+================+=================+==========================+
    | **a**      | a / c          | -(a/d)j         | special rules            |
    +------------+----------------+-----------------+--------------------------+
    | **bj**     | (b/c)j         | b/d             | special rules            |
    +------------+----------------+-----------------+--------------------------+
    | **a + bj** | (a/c) + (b/c)j | b/d - (a/d)j    | special rules            |
    +------------+----------------+-----------------+--------------------------+

    In general, for complex floating-point operands,
    real-valued floating-point special cases
    must independently apply to the real and imaginary component operations
    involving real numbers as described in the above table.

    When ``a``, ``b``, ``c``, or ``d`` are all finite numbers
    (i.e., a value other than ``NaN``, ``+infinity``, or ``-infinity``),
    division of complex floating-point operands should be computed as
    if calculated according to the textbook formula for complex number division

    .. math::
       \frac{a + bj}{c + dj} = \frac{(ac + bd) + (bc - ad)j}{c^2 + d^2}

    When at least one of ``a``, ``b``, ``c``, or ``d`` is ``NaN``,
    ``+infinity``, or ``-infinity``,

    - If ``a``, ``b``, ``c``, and ``d`` are all ``NaN``, the result is ``NaN + NaN j``.
    - In the remaining cases, the result is implementation dependent.

    .. note::
       For complex floating-point operands, the results of special cases
       may be implementation dependent depending on how an implementation
       chooses to model complex numbers and complex infinity (e.g.,
       complex plane versus Riemann sphere).
       For those implementations following C99 and its one-infinity model,
       when at least one component is infinite, even if the other component is ``NaN``,
       the complex value is infinite, and the usual arithmetic rules do not apply to
       complex-complex division.
       In the interest of performance, other implementations may want
       to avoid the complex branching logic necessary
       to implement the one-infinity model
       and choose to implement all complex-complex division
       according to the textbook formula.
       Accordingly, special case behavior is unlikely
       to be consistent across implementations.

    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.divide.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Parameters
    ----------
    x1
        dividend input array. Should have a numeric data type.
    x2
        divisor input array. Must be compatible with x1 (see Broadcasting). Should have
        a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        floating-point data type determined by Type Promotion Rules.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x1 = ivy.array([2., 7., 9.])
    >>> x2 = ivy.array([3., 4., 0.6])
    >>> y = ivy.divide(x1, x2)
    >>> print(y)
    ivy.array([0.667, 1.75, 15.])

    With mixed :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x1 = ivy.array([5., 6., 9.])
    >>> x2 = ivy.native_array([2., 2., 2.])
    >>> y = ivy.divide(x1, x2)
    >>> print(y)
    ivy.array([2.5, 3., 4.5])

    With :class:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([12., 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
    >>> x2 = ivy.Container(a=ivy.array([1., 2.3, 3]), b=ivy.array([2.4, 3., 2.]))
    >>> y = ivy.divide(x1, x2)
    >>> print(y)
    {
        a: ivy.array([12., 1.52, 2.1]),
        b: ivy.array([1.25, 0.333, 0.45])
    }

    With mixed :class:`ivy.Container` and :class:`ivy.Array` inputs:

    >>> x1 = ivy.Container(a=ivy.array([12., 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
    >>> x2 = ivy.array([4.3, 3., 5.])
    >>> y = ivy.divide(x1, x2)
    {
        a: ivy.array([2.79, 1.17, 1.26]),
        b: ivy.array([0.698, 0.333, 0.18])
    }
    """
    return ivy.current_backend(x1, x2).divide(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def equal(
    x1: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
    x2: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the truth value of x1_i == x2_i for each element x1_i of the input array x1
    with the respective element x2_i of the input array x2.

    Parameters
    ----------
    x1
        first input array. May have any data type.
    x2
        second input array. Must be compatible with x1 (with Broadcasting). May have any
        data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type of bool.


    **Special cases**

    For real-valued floating-point operands,

    - If ``x1_i`` is ``NaN`` or ``x2_i`` is ``NaN``, the result is ``False``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``+infinity``,
      the result is ``True``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``-infinity``,
      the result is ``True``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is either ``+0`` or ``-0``,
      the result is ``True``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is either ``+0`` or ``-0``,
      the result is ``True``.
    - If ``x1_i`` is a finite number, ``x2_i`` is a finite number,
      and ``x1_i`` equals ``x2_i``, the result is ``True``.
    - In the remaining cases, the result is ``False``.

    For complex floating-point operands, let ``a = real(x1_i)``,
    ``b = imag(x1_i)``, ``c = real(x2_i)``, ``d = imag(x2_i)``, and

    - If ``a``, ``b``, ``c``, or ``d`` is ``NaN``, the result is ``False``.
    - In the remaining cases, the result is the logical AND of the equality
      comparison between the real values ``a`` and ``c`` (real components) and
      between the real values ``b`` and ``d`` (imaginary components),
      as described above for real-valued floating-point operands
      (i.e., ``a == c AND b == d``).

    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.equal.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x1 = ivy.array([2., 7., 9.])
    >>> x2 = ivy.array([1., 7., 9.])
    >>> y = ivy.equal(x1, x2)
    >>> print(y)
    ivy.array([False, True, True])

    With mixed :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x1 = ivy.array([5, 6, 9])
    >>> x2 = ivy.native_array([2, 6, 2])
    >>> y = ivy.equal(x1, x2)
    >>> print(y)
    ivy.array([False, True, False])

    With :class:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([12, 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
    >>> x2 = ivy.Container(a=ivy.array([12, 2.3, 3]), b=ivy.array([2.4, 3., 2.]))
    >>> y = ivy.equal(x1, x2)
    >>> print(y)
    {
        a: ivy.array([True, False, False]),
        b: ivy.array([False, False, False])
    }

    With mixed :class:`ivy.Container` and :class:`ivy.Array` inputs:

    >>> x1 = ivy.Container(a=ivy.array([12., 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
    >>> x2 = ivy.array([3., 1., 0.9])
    >>> y = ivy.equal(x1, x2)
    >>> print(y)
    {
        a: ivy.array([False, False, False]),
        b: ivy.array([True, True, True])
    }
    """
    return ivy.current_backend(x1, x2).equal(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def exp(
    x: Union[ivy.Array, ivy.NativeArray, Number],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation to the exponential function,
    having domain ``[-infinity, +infinity]`` and codomain ``[+0, +infinity]``, for each
    element ``x_i`` of the input array ``x`` (``e`` raised to the power of ``x_i``,
    where ``e`` is the base of the natural logarithm).

    .. note::
        For complex floating-point operands, ``exp(conj(x))`` must
        equal ``conj(exp(x))``.

    .. note::
        The exponential function is an entire function in
        the complex plane and has no branch cuts.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``1``.
    - If ``x_i`` is ``-0``, the result is ``1``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``+0``.

    For complex floating-point operands,
    let ``a = real(x_i)``, ``b = imag(x_i)``, and

    - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``,
      the result is ``1 + 0j``.
    - If ``a`` is a finite number and ``b`` is ``+infinity``,
      the result is ``NaN + NaN j``.
    - If ``a`` is a finite number and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+0``,
      the result is ``infinity + 0j``.
    - If ``a`` is ``-infinity`` and ``b`` is a finite number,
      the result is ``+0 * cis(b)``.
    - If ``a`` is ``+infinity`` and ``b`` is a nonzero finite number,
      the result is ``+infinity * cis(b)``.
    - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``,
      the result is ``0 + 0j`` (signs of real and imaginary components are unspecified).
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``infinity + NaN j`` (sign of real component is unspecified).
    - If ``a`` is ``-infinity`` and ``b`` is ``NaN``,
      the result is ``0 + 0j`` (signs of real and imaginary components are unspecified).
    - If ``a`` is ``+infinity`` and ``b`` is ``NaN``,
      the result is ``infinity + NaN j`` (sign of real component is unspecified).
    - If ``a`` is ``NaN`` and ``b`` is ``+0``,
      the result is ``NaN + 0j``.
    - If ``a`` is ``NaN`` and ``b`` is not equal to ``0``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.

    where ``cis(v)`` is ``cos(v) + sin(v)*1j``.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the evaluated exponential function result for each element
        in ``x``. The returned array must have a floating-point data type determined by
        :ref:`type-promotion`.

    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.exp.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:Number:

    >>> x = 3
    >>> y = ivy.exp(x)
    >>> print(y)
    ivy.array(20.08553692)

    With :class:`ivy.Array` input:

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.exp(x)
    >>> print(y)
    ivy.array([ 2.71828175,  7.38905621, 20.08553696])

    With nested inputs in :class:`ivy.Array`:

    >>> x = ivy.array([[-5.67], [ivy.nan], [0.567]])
    >>> y = ivy.exp(x)
    >>> print(y)
    ivy.array([[0.00344786],
           [       nan],
           [1.76297021]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0., 4., 2.])
    >>> y = ivy.exp(x)
    >>> print(y)
    ivy.array([ 1.        , 54.59814835,  7.38905621])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=3.1, b=ivy.array([3.2, 1.]))
    >>> y = ivy.exp(x)
    >>> print(y)
    {
        a: ivy.array(22.197948),
        b: ivy.array([24.53253174, 2.71828175])
    }
    """
    return ivy.current_backend(x).exp(x, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def imag(
    val: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the imaginary part of a complex number for each element
    ``x_i`` of the input array ``val``.

    Parameters
    ----------
    val
        input array. Should have a complex floating-point data type.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns an array with the imaginary part of complex numbers.
        The returned arrau must have a floating-point data type determined by
        the precision of ``val`` (e.g., if ``val`` is ``complex64``,
        the returned array must be ``float32``).

    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.imag.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> b = ivy.array(np.array([1+2j, 3+4j, 5+6j]))
    >>> b
    ivy.array([1.+2.j, 3.+4.j, 5.+6.j])
    >>> ivy.imag(b)
    ivy.array([2., 4., 6.])
    """
    return ivy.current_backend(val).imag(val, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def angle(
    z: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    deg: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate Element-wise the angle for an array of complex numbers(x+yj).

    Parameters
    ----------
    z
        Array-like input.
    deg
        optional bool.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns an array of angles for each complex number in the input.
        If deg is False(default), angle is calculated in radian and if
        deg is True, then angle is calculated in degrees.

    Examples
    --------
    >>> ivy.set_backend('tensorflow')
    >>> z = ivy.array([-1 + 1j, -2 + 2j, 3 - 3j])
    >>> z
    ivy.array([-1.+1.j, -2.+2.j,  3.-3.j])
    >>> ivy.angle(z)
    ivy.array([ 2.35619449,  2.35619449, -0.78539816])
    >>> ivy.set_backend('numpy')
    >>> ivy.angle(z,deg=True)
    ivy.array([135., 135., -45.])
    """
    return ivy.current_backend(z).angle(z, deg=deg, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def gcd(
    x1: Union[ivy.Array, ivy.NativeArray, int, list, tuple],
    x2: Union[ivy.Array, ivy.NativeArray, int, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the greatest common divisor of |x1| and |x2|.

    Parameters
    ----------
    x1
        First array-like input.
    x2
        Second array-input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Element-wise gcd of |x1| and |x2|.

    Examples
    --------
    >>> x1 = ivy.array([1, 2, 3])
    >>> x2 = ivy.array([4, 5, 6])
    >>> ivy.gcd(x1, x2)
    ivy.array([1.,    1.,   3.])
    >>> x1 = ivy.array([1, 2, 3])
    >>> ivy.gcd(x1, 10)
    ivy.array([1.,   2.,  1.])
    """
    return ivy.current_backend(x1, x2).gcd(x1, x2, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def exp2(
    x: Union[ivy.Array, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate 2**p for all p in the input array.

    Parameters
    ----------
    x
        Array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Element-wise 2 to the power x. This is a scalar if x is a scalar.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> ivy.exp2(x)
    ivy.array([2.,    4.,   8.])
    >>> x = [5, 6, 7]
    >>> ivy.exp2(x)
    ivy.array([32.,   64.,  128.])
    """
    return ivy.current_backend(x).exp2(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def expm1(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation to ``exp(x)-1``, having
    domain ``[-infinity, +infinity]`` and codomain ``[-1, +infinity]``, for each element
    ``x_i`` of the input array ``x``.

    .. note::
       The purpose of this function is to calculate ``exp(x)-1.0`` more accurately when
       ``x`` is close to zero. Accordingly, conforming implementations should avoid
       implementing this function as simply ``exp(x)-1.0``. See FDLIBM, or some other
       IEEE 754-2019 compliant mathematical library, for a potential reference
       implementation.

    .. note::
        For complex floating-point operands, ``expm1(conj(x))``
        must equal ``conj(expm1(x))``.

    .. note::
        The exponential function is an entire function
        in the complex plane and has no branch cuts.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-1``.

    For complex floating-point operands,
    let ``a = real(x_i)``, ``b = imag(x_i)``, and

    - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``,
      the result is ``0 + 0j``.
    - If ``a`` is a finite number and ``b`` is ``+infinity``,
      the result is ``NaN + NaN j``.
    - If ``a`` is a finite number and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+0``,
      the result is ``+infinity + 0j``.
    - If ``a`` is ``-infinity`` and ``b`` is a finite number,
      the result is ``+0 * cis(b) - 1.0``.
    - If ``a`` is ``+infinity`` and ``b`` is a nonzero finite number,
      the result is ``+infinity * cis(b) - 1.0``.
    - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``,
      the result is ``-1 + 0j`` (sign of imaginary component is unspecified).
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``infinity + NaN j`` (sign of real component is unspecified).
    - If ``a`` is ``-infinity`` and ``b`` is ``NaN``,
      the result is ``-1 + 0j`` (sign of imaginary component is unspecified).
    - If ``a`` is ``+infinity`` and ``b`` is ``NaN``,
      the result is ``infinity + NaN j`` (sign of real component is unspecified).
    - If ``a`` is ``NaN`` and ``b`` is ``+0``,
      the result is ``NaN + 0j``.
    - If ``a`` is ``NaN`` and ``b`` is not equal to ``0``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.

    where ``cis(v)`` is ``cos(v) + sin(v)*1j``.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the evaluated result for each element in ``x``. The returned
        array must have a floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.expm1.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([[0, 5, float('-0'), ivy.nan]])
    >>> ivy.expm1(x)
    ivy.array([[  0., 147.,  -0.,  nan]])

    >>> x = ivy.array([ivy.inf, 1, float('-inf')])
    >>> y = ivy.zeros(3)
    >>> ivy.expm1(x, out=y)
    ivy.array([  inf,  1.72, -1.  ])

    With :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([-1, 0,]),
    ...                   b=ivy.array([10, 1]))
    >>> ivy.expm1(x)
    {
        a: ivy.array([-0.632, 0.]),
        b: ivy.array([2.20e+04, 1.72e+00])
    }
    """
    return ivy.current_backend(x).expm1(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def floor(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Round each element ``x_i`` of the input array ``x`` to the greatest (i.e.,
    closest to ``+infinity``) integer-valued number that is not greater than ``x_i``.

    **Special cases**

    - If ``x_i`` is already integer-valued, the result is ``x_i``.

    For floating-point operands,

    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``NaN``, the result is ``NaN``.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rounded result for each element in ``x``. The returned
        array must have the same data type as ``x``.


    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.floor.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([2,3,4])
    >>> y = ivy.floor(x)
    >>> print(y)
    ivy.array([2, 3, 4])

    >>> x = ivy.array([1.5, -5.5, 0, -1, -0])
    >>> y = ivy.zeros(5)
    >>> ivy.floor(x, out=y)
    >>> print(y)
    ivy.array([ 1., -6.,  0., -1.,  0.])

    >>> x = ivy.array([[1.1, 2.2, 3.3], [-4.4, -5.5, -6.6]])
    >>> ivy.floor(x, out=x)
    >>> print(x)
    ivy.array([[ 1.,  2.,  3.],
               [-5., -6., -7.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1.5, -2.4]),
    ...                   b=ivy.array([3.4, -4.2, -0, -1.2]))
    >>> y = ivy.floor(x)
    >>> print(y)
    {
        a: ivy.array([0., 1., -3.]),
        b: ivy.array([3., -5., 0., -2.])
    }

    """
    return ivy.current_backend(x).floor(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def floor_divide(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""
    Round the result of dividing each element x1_i of the input array x1 by the
    respective element x2_i of the input array x2 to the greatest (i.e., closest to
    +infinity) integer-value number that is not greater than the division result.

    .. note::
        For input arrays which promote to an integer data type,
        the result of division by zero is unspecified and thus implementation-defined.

    **Special cases**

    .. note::
       Floor division was introduced in Python via
       `PEP 238 <https://www.python.org/dev/peps/pep-0238/>`_
       with the goal to disambiguate "true division"
       (i.e., computing an approximation to the mathematical operation of division)
       from "floor division" (i.e., rounding the result
       of division toward negative infinity).
       The former was computed when one of the operands was a ``float``,
       while the latter was computed when both operands were ``int``\s.
       Overloading the ``/`` operator to support both behaviors led to
       subtle numerical bugs when integers are possible, but not expected.

       To resolve this ambiguity, ``/`` was designated for true division, and ``//``
       was designated for floor division. Semantically, floor division was
       `defined
       <https://www.python.org/dev/peps/pep-0238/#semantics-of-floor-division>`_
       as equivalent to ``a // b == floor(a/b)``;
       however, special floating-point cases were left ill-defined.

       Accordingly, floor division is not implemented consistently
       across array libraries for some of the special cases documented below.
       Namely, when one of the operands is ``infinity``,
       libraries may diverge with some choosing
       to strictly follow ``floor(a/b)`` and others choosing to pair ``//`` with ``%``
       according to the relation ``b = a % b + b * (a // b)``.
       The special cases leading to divergent behavior are documented below.

       This specification prefers floor division to match ``floor(divide(x1, x2))``
       in order to avoid surprising and unexpected results; however,
       array libraries may choose to more strictly follow Python behavior.

    For floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i``
      is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
    - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i``
      is either ``+0`` or ``-0``, the result is ``NaN``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``,
      the result is ``+0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``,
      the result is ``-0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``,
      the result is ``-0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``,
      the result is ``+0``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``,
      the result is ``+infinity``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``,
      the result is ``-infinity``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``,
      the result is ``-infinity``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``,
      the result is ``+infinity``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``+infinity``.
      (**note**: libraries may return ``NaN`` to match Python behavior.)
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative
      (i.e., less than ``0``) finite number, the result is ``-infinity``.
      (**note**: libraries may return ``NaN`` to match Python behavior.)
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``-infinity``.
      (**note**: libraries may return ``NaN`` to match Python behavior.)
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative
      (i.e., less than ``0``) finite number, the result is ``+infinity``.
      (**note**: libraries may return ``NaN`` to match Python behavior.)
    - If ``x1_i`` is a positive (i.e., greater than ``0``)
      finite number and ``x2_i`` is ``+infinity``, the result is ``+0``.
    - If ``x1_i`` is a positive (i.e., greater than ``0``)
      finite number and ``x2_i`` is ``-infinity``, the result is ``-0``.
      (**note**: libraries may return ``-1.0`` to match Python behavior.)
    - If ``x1_i`` is a negative (i.e., less than ``0``)
      finite number and ``x2_i`` is ``+infinity``, the result is ``-0``.
      (**note**: libraries may return ``-1.0`` to match Python behavior.)
    - If ``x1_i`` is a negative (i.e., less than ``0``)
      finite number and ``x2_i`` is ``-infinity``, the result is ``+0``.
    - If ``x1_i`` and ``x2_i`` have the same mathematical sign and
      are both nonzero finite numbers, the result has a positive mathematical sign.
    - If ``x1_i`` and ``x2_i`` have different mathematical signs and
      are both nonzero finite numbers, the result has a negative mathematical sign.
    - In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``,
      nor ``NaN`` is involved, the quotient must be computed and rounded to
      the greatest (i.e., closest to `+infinity`) representable integer-value
      number that is not greater than the division result.
      If the magnitude is too large to represent, the operation overflows and
      the result is an ``infinity`` of appropriate mathematical sign.
      If the magnitude is too small to represent, the operation underflows and
      the result is a zero of appropriate mathematical sign.

    Parameters
    ----------
    x1
        first input array. Must have a numeric data type.
    x2
        second input array. Must be compatible with x1 (with Broadcasting). Must have a
        numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        numeric data type.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.floor_divide.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments


    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x1 = ivy.array([13., 7., 8.])
    >>> x2 = ivy.array([3., 2., 7.])
    >>> y = ivy.floor_divide(x1, x2)
    >>> print(y)
    ivy.array([4., 3., 1.])

    With mixed :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x1 = ivy.array([3., 4., 5.])
    >>> x2 = ivy.native_array([5., 2., 1.])
    >>> y = ivy.floor_divide(x1, x2)
    >>> print(y)
    ivy.array([0., 2., 5.])

    With :class:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([4., 5., 6.]), b=ivy.array([7., 8., 9.]))
    >>> x2 = ivy.Container(a=ivy.array([5., 4., 2.5]), b=ivy.array([2.3, 3.7, 5]))
    >>> y = ivy.floor_divide(x1, x2)
    >>> print(y)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 2., 1.])
    }

    With mixed :class:`ivy.Container` and :class:`ivy.Array` inputs:

    >>> x1 = ivy.Container(a=ivy.array([4., 5., 6.]), b=ivy.array([7., 8., 9.]))
    >>> x2 = ivy.array([2., 2., 2.])
    >>> y = ivy.floor_divide(x1, x2)
    >>> print(y)
    {
        a: ivy.array([2., 2., 3.]),
        b: ivy.array([3., 4., 4.])
    }
    """
    return ivy.current_backend(x1, x2).floor_divide(x1, x2, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def fmin(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Compute the element-wise minimums of two arrays. Differs from ivy.minimum in the
    case where one of the elements is NaN. ivy.minimum returns the NaN element while
    ivy.fmin returns the non-NaN element.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with element-wise minimums.

    Examples
    --------
    >>> x1 = ivy.array([2, 3, 4])
    >>> x2 = ivy.array([1, 5, 2])
    >>> ivy.fmin(x1, x2)
    ivy.array([1, 3, 2])

    >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
    >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
    >>> ivy.fmin(x1, x2)
    ivy.array([ 0.,  0., nan])
    """
    return ivy.current_backend(x1, x2).fmin(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def greater(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the truth value of x1_i < x2_i for each element x1_i of the input array x1
    with the respective element x2_i of the input array x2.

    Parameters
    ----------
    x1
        Input array.
    x2
        Input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type of bool.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.greater.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.greater(ivy.array([1,2,3]),ivy.array([2,2,2]))
    >>> print(x)
    ivy.array([False, False,  True])

    >>> x = ivy.array([[[1.1], [3.2], [-6.3]]])
    >>> y = ivy.array([[8.4], [2.5], [1.6]])
    >>> ivy.greater(x, y, out=x)
    >>> print(x)
    ivy.array([[[False],[True],[False]]])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.native_array([4, 5, 0])
    >>> z = ivy.greater(x, y)
    >>> print(z)
    ivy.array([False,False,True])

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([[5.1, 2.3, -3.6]])
    >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),
    ...                   b=ivy.array([[5.], [6.], [7.]]))
    >>> z = ivy.greater(x, y)
    >>> print(z)
    {
        a: ivy.array([[True, False, False],
                      [True, False, False],
                      [False, False, False]]),
        b: ivy.array([[True, False, False],
                      [False, False, False],
                      [False, False, False]])
    }

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([4, 5, 6]),
    ...                   b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                   b=ivy.array([5, 6, 7]))
    >>> z = ivy.greater(x, y)
    >>> print(z)
    {
        a: ivy.array([True, True, True]),
        b: ivy.array([False, False, False])
    }
    """
    return ivy.current_backend(x1, x2).greater(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def greater_equal(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the truth value of x1_i >= x2_i for each element x1_i of the input array x1
    with the respective element x2_i of the input array x2.

    Parameters
    ----------
    x1
        first input array. May have any data type.
    x2
        second input array. Must be compatible with x1 (with Broadcasting). May have any
        data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type of bool.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.greater_equal.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.greater_equal(ivy.array([1,2,3]),ivy.array([2,2,2]))
    >>> print(x)
    ivy.array([False,True,True])

    >>> x = ivy.array([[10.1, 2.3, -3.6]])
    >>> y = ivy.array([[4.8], [5.2], [6.1]])
    >>> shape = (3,3)
    >>> fill_value = False
    >>> z = ivy.full(shape, fill_value)
    >>> ivy.greater_equal(x, y, out=z)
    >>> print(z)
    ivy.array([[True,False,False],[True,False,False],[True,False,False]])

    >>> x = ivy.array([[[1.1], [3.2], [-6.3]]])
    >>> y = ivy.array([[8.4], [2.5], [1.6]])
    >>> ivy.greater_equal(x, y, out=x)
    >>> print(x)
    ivy.array([[[False],[True],[False]]])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.native_array([4, 5, 0])
    >>> z = ivy.greater_equal(x, y)
    >>> print(z)
    ivy.array([False,False,True])

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([[5.1, 2.3, -3.6]])
    >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]), b=ivy.array([[5.], [6.], [7.]])) # noqa
    >>> z = ivy.greater_equal(x, y)
    >>> print(z)
    {
        a:ivy.array([[True,False,False],[True,False,False],[False,False,False]]),
        b:ivy.array([[True,False,False],[False,False,False],[False,False,False]])
    }

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([1, 2, 3]),b=ivy.array([5, 6, 7]))
    >>> z = ivy.greater_equal(x, y)
    >>> print(z)
    {
        a:ivy.array([True,True,True]),
        b:ivy.array([False,False,False])
    }
    """
    return ivy.current_backend(x1, x2).greater_equal(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def less_equal(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the truth value of x1_i <= x2_i for each element x1_i of the input array x1
    with the respective element x2_i of the input array x2.

    Parameters
    ----------
    x1
        first input array. May have any data type.
    x2
        second input array. Must be compatible with x1 (with Broadcasting). May have any
        data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
     ret
        an array containing the element-wise results. The returned array must have a
        data type of bool.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.less_equal.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.less_equal(ivy.array([1,2,3]),ivy.array([2,2,2]))
    >>> print(x)
    ivy.array([True, True,  False])

    >>> x = ivy.array([[10.1, 2.3, -3.6]])
    >>> y = ivy.array([[4.8], [5.2], [6.1]])
    >>> shape = (3,3)
    >>> fill_value = False
    >>> z = ivy.full(shape, fill_value)
    >>> ivy.less_equal(x, y, out=z)
    >>> print(z)
    ivy.array([[False, True, True],
       [ False, True, True],
       [ False, True, True]])

    >>> x = ivy.array([[[1.1], [3.2], [-6.3]]])
    >>> y = ivy.array([[8.4], [2.5], [1.6]])
    >>> ivy.less_equal(x, y, out=x)
    >>> print(x)
    ivy.array([[[True],[False],[True]]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([1, 2, 3]),b=ivy.array([5, 6, 7]))
    >>> z = ivy.less_equal(x, y)
    >>> print(z)
    {
        a: ivy.array([False, False, False]),
        b: ivy.array([True, True, True])
    }
    """
    return ivy.current_backend(x1, x2).less_equal(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def multiply(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""
    Calculate the product for each element x1_i of the input array x1 with the
    respective element x2_i of the input array x2.

    .. note::
       Floating-point multiplication is not always associative due to finite precision.

    **Special Cases**

    For real-valued floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and
      ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
    - If ``x1_i`` is either ``+0`` or ``-0`` and
      ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
    - If ``x1_i`` and ``x2_i`` have the same mathematical sign,
      the result has a positive mathematical sign, unless the result is ``NaN``.
      If the result is ``NaN``, the "sign" of ``NaN`` is implementation-defined.
    - If ``x1_i`` and ``x2_i`` have different mathematical signs,
      the result has a negative mathematical sign,
      unless the result is ``NaN``. If the result is ``NaN``,
      the "sign" of ``NaN`` is implementation-defined.
    - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and
      ``x2_i`` is either ``+infinity`` or ``-infinity``,
      the result is a signed infinity with the mathematical sign determined by
      the rule already stated above.
    - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i``
      is a nonzero finite number, the result is a signed infinity with
      the mathematical sign determined by the rule already stated above.
    - If ``x1_i`` is a nonzero finite number and ``x2_i``
      is either ``+infinity`` or ``-infinity``, the result is a signed infinity with
      the mathematical sign determined by the rule already stated above.
    - In the remaining cases, where neither ``infinity`` nor ``NaN``
      is involved, the product must be computed and rounded to the nearest
      representable value according to IEEE 754-2019 and a supported
      rounding mode. If the magnitude is too large to represent,
      the result is an `infinity` of appropriate mathematical sign.
      If the magnitude is too small to represent, the result is a zero of
      appropriate mathematical sign.

    For complex floating-point operands, multiplication is defined according to the
    following table. For real components ``a`` and ``c`` and
    imaginary components ``b`` and ``d``,

    +------------+----------------+-----------------+--------------------------+
    |            | c              | dj              | c + dj                   |
    +============+================+=================+==========================+
    | **a**      | a * c          | (a*d)j          | (a*c) + (a*d)j           |
    +------------+----------------+-----------------+--------------------------+
    | **bj**     | (b*c)j         | -(b*d)          | -(b*d) + (b*c)j          |
    +------------+----------------+-----------------+--------------------------+
    | **a + bj** | (a*c) + (b*c)j | -(b*d) + (a*d)j | special rules            |
    +------------+----------------+-----------------+--------------------------+

    In general, for complex floating-point operands, real-valued floating-point
    special cases must independently apply to the real and imaginary component
    operations involving real numbers as described in the above table.

    When ``a``, ``b``, ``c``, or ``d`` are all finite numbers
    (i.e., a value other than ``NaN``, ``+infinity``, or ``-infinity``),
    multiplication of complex floating-point operands should be computed
    as if calculated according to the textbook formula for complex number multiplication

    .. math::
       (a + bj) \cdot (c + dj) = (ac - bd) + (bc + ad)j

    When at least one of ``a``, ``b``, ``c``, or ``d`` is ``NaN``,
    ``+infinity``, or ``-infinity``,

    - If ``a``, ``b``, ``c``, and ``d`` are all ``NaN``,
      the result is ``NaN + NaN j``.
    - In the remaining cases, the result is implementation dependent.

    .. note::
       For complex floating-point operands, the results of special cases may be
       implementation dependent depending on how an implementation chooses
       to model complex numbers and complex infinity
       (e.g., complex plane versus Riemann sphere).
       For those implementations following C99 and its one-infinity model,
       when at least one component is infinite,
       even if the other component is ``NaN``,
       the complex value is infinite, and the usual arithmetic
       rules do not apply to complex-complex multiplication.
       In the interest of performance, other implementations
       may want to avoid the complex branching logic necessary
       to implement the one-infinity model and choose to implement
       all complex-complex multiplication according to the textbook formula.
       Accordingly, special case behavior is unlikely
       to be consistent across implementations.

    Parameters
    ----------
    x1
        first input array. Should have a numeric data type.

    x2
        second input array. Must be compatible with ``x1``
        (see :ref'`broadcasting`). Should have a numeric data type

    out
        optional output array, for writing the array result to.
        It must have a shape that the inputs broadcast to.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.multiply.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Returns
    -------
    ret
        an array containing the element-wise products. The returned array must have a
        data type determined by :ref:`Type Promotion Rules`.

    Examples
    --------
    With :code:`ivy.Array` inputs:

    >>> x1 = ivy.array([3., 5., 7.])
    >>> x2 = ivy.array([4., 6., 8.])
    >>> y = ivy.multiply(x1, x2)
    >>> print(y)
    ivy.array([12., 30., 56.])

    With :code:`ivy.NativeArray` inputs:

    >>> x1 = ivy.native_array([1., 3., 9.])
    >>> x2 = ivy.native_array([4., 7.2, 1.])
    >>> y = ivy.multiply(x1, x2)
    >>> print(y)
    ivy.array([ 4. , 21.6,  9. ])

    With mixed :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x1 = ivy.array([8., 6., 7.])
    >>> x2 = ivy.native_array([1., 2., 3.])
    >>> y = ivy.multiply(x1, x2)
    >>> print(y)
    ivy.array([ 8., 12., 21.])

    With :code:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([12.,4.,6.]), b=ivy.array([3.,1.,5.]))
    >>> x2 = ivy.Container(a=ivy.array([1.,3.,4.]), b=ivy.array([3.,3.,2.]))
    >>> y = ivy.multiply(x1, x2)
    >>> print(y)
    {
        a: ivy.array([12.,12.,24.]),
        b: ivy.array([9.,3.,10.])
    }

    With mixed :code:`ivy.Container` and :code:`ivy.Array` inputs:

    >>> x1 = ivy.Container(a=ivy.array([3., 4., 5.]), b=ivy.array([2., 2., 1.]))
    >>> x2 = ivy.array([1.,2.,3.])
    >>> y = ivy.multiply(x1, x2)
    >>> print(y)
    {
        a: ivy.array([3.,8.,15.]),
        b: ivy.array([2.,4.,3.])
    }
    """
    return ivy.current_backend(x1, x2).multiply(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def isfinite(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Test each element ``x_i`` of the input array ``x`` to determine if finite (i.e.,
    not ``NaN`` and not equal to positive or negative infinity).

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing test results. An element ``out_i`` is ``True`` if ``x_i`` is
        finite and ``False`` otherwise. The returned array must have a data type of
        ``bool``.


    **Special Cases**

    For real-valued floating-point operands,

    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``False``.
    - if ``x_i`` is ``NaN``, the result is ``False``.
    - if ``x_i`` is a finite number, the result is ``True``.

    For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``,
    and

    - If ``a`` is ``NaN`` or ``b`` is ``NaN``, the result is ``False``.
    _ If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is any value,
      the result is ``False``.
    - If ``a`` is any value and ``b`` is either ``+infinity`` or ``-infinity``,
      the result is ``False``.
    - If ``a`` is a finite number and ``b`` is a finite number, the result is ``True``.

    This method conforms to the
    `Array API Standard<https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the `docstring
    <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.isfinite.html>`
    _ in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, ivy.nan, -ivy.inf, float('inf')])
    >>> y = ivy.isfinite(x)
    >>> print(y)
    ivy.array([ True, False, False, False])

    >>> x = ivy.array([0, ivy.nan, -ivy.inf])
    >>> y = ivy.zeros(3)
    >>> ivy.isfinite(x, out=y)
    >>> print(y)
    ivy.array([ True, False, False])

    >>> x = ivy.array([[9, float('-0')], [ivy.nan, ivy.inf]])
    >>> ivy.isfinite(x, out=x)
    >>> print(x)
    ivy.array([[ True,  True],
        [False, False]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 999999999999]),
    ...                   b=ivy.array([float('-0'), ivy.nan]))
    >>> y = ivy.isfinite(x)
    >>> print(y)
    {
        a: ivy.array([True, True]),
        b: ivy.array([True, False])
    }
    """
    return ivy.current_backend(x).isfinite(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def isinf(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    detect_positive: bool = True,
    detect_negative: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Test each element x_i of the input array x to determine if equal to positive or
    negative infinity.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    detect_positive
        if ``True``, positive infinity is detected.
    detect_negative
        if ``True``, negative infinity is detected.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing test results. An element out_i is True if x_i is either
        positive or negative infinity and False otherwise. The returned array must have
        a data type of bool.


    **Special Cases**

    For real-valued floating-point operands,

    - If x_i is either +infinity or -infinity, the result is ``True``.
    - In the remaining cases, the result is ``False``.

    For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``,
    and

    - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is any value
      (including ``NaN``), the result is ``True``.
    - If ``a`` is either a finite number or ``NaN`` and ``b`` is either ``+infinity``
      or ``-infinity``, the result is ``True``.
    - In the remaining cases, the result is ``False``.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.isinf.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> z = ivy.isinf(x)
    >>> print(z)
    ivy.array([False, False, False])

    >>> x = ivy.array([[1.1, 2.3, -3.6]])
    >>> z = ivy.isinf(x)
    >>> print(z)
    ivy.array([[False, False, False]])

    >>> x = ivy.array([[[1.1], [float('inf')], [-6.3]]])
    >>> z = ivy.isinf(x)
    >>> print(z)
    ivy.array([[[False],
            [True],
            [False]]])

    >>> x = ivy.array([[-float('inf'), float('inf'), 0.0]])
    >>> z = ivy.isinf(x)
    >>> print(z)
    ivy.array([[ True,  True, False]])

    >>> x = ivy.zeros((3, 3))
    >>> z = ivy.isinf(x)
    >>> print(z)
    ivy.array([[False, False, False],
       [False, False, False],
       [False, False, False]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1, -float('inf'), 1.23]),
    ...                   b=ivy.array([float('inf'), 3.3, -4.2]))
    >>> z = ivy.isinf(x)
    >>> print(z)
    {
        a: ivy.array([False, True, False]),
        b: ivy.array([True, False, False])
    }

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1, -float('inf'), 1.23]),
    ...                   b=ivy.array([float('inf'), 3.3, -4.2]))
    >>> x.isinf()
    {
        a: ivy.array([False, True, False]),
        b: ivy.array([True, False, False])
    }
    """
    return ivy.current_backend(x).isinf(
        x, detect_positive=detect_positive, detect_negative=detect_negative, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def isnan(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Test each element ``x_i`` of the input array ``x`` to determine whether the
    element is ``NaN``.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing test results. An element ``out_i`` is ``True`` if ``x_i`` is
        ``NaN`` and ``False`` otherwise. The returned array should have a data type of
        ``bool``.


    **Special Cases**

    For real-valued floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``True``.
    - In the remaining cases, the result is ``False``.

    For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``,
    and

    - If ``a`` or ``b`` is ``NaN``, the result is ``True``.
    - In the remaining cases, the result is ``False``.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.isnan.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> z = ivy.isnan(x)
    >>> print(z)
    ivy.array([False, False, False])

    >>> x = ivy.array([[1.1, 2.3, -3.6]])
    >>> z = ivy.isnan(x)
    >>> print(z)
    ivy.array([[False, False, False]])

    >>> x = ivy.array([[[1.1], [float('inf')], [-6.3]]])
    >>> z = ivy.isnan(x)
    >>> print(z)
    ivy.array([[[False],
                [False],
                [False]]])

    >>> x = ivy.array([[-float('nan'), float('nan'), 0.0]])
    >>> z = ivy.isnan(x)
    >>> print(z)
    ivy.array([[ True,  True, False]])

    >>> x = ivy.array([[-float('nan'), float('inf'), float('nan'), 0.0]])
    >>> z = ivy.isnan(x)
    >>> print(z)
    ivy.array([[ True, False,  True, False]])

    >>> x = ivy.zeros((3, 3))
    >>> z = ivy.isnan(x)
    >>> print(z)
    ivy.array([[False, False, False],
       [False, False, False],
       [False, False, False]])


    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1, -float('nan'), 1.23]),
    ...                   b=ivy.array([float('nan'), 3.3, -4.2]))
    >>> z = ivy.isnan(x)
    >>> print(z)
    {
        a: ivy.array([False, True, False]),
        b: ivy.array([True, False, False])
    }
    """
    return ivy.current_backend(x).isnan(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def less(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the truth value of ``x1_i < x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a numeric data type.
    x2
        second input array. Must be compatible with ``x1`` (see  ref:`broadcasting`).
        Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type of ``bool``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.less(ivy.array([1,2,3]),ivy.array([2,2,2]))
    >>> print(x)
    ivy.array([ True, False, False])


    >>> x = ivy.array([[[1.1], [3.2], [-6.3]]])
    >>> y = ivy.array([[8.4], [2.5], [1.6]])
    >>> ivy.less(x, y, out=x)
    >>> print(x)
    ivy.array([[[True],[False],[True]]])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.native_array([4, 5, 0])
    >>> z = ivy.less(x, y)
    >>> print(z)
    ivy.array([ True,  True, False])

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([[5.1, 2.3, -3.6]])
    >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),
    ...                   b=ivy.array([[5.], [6.], [7.]]))
    >>> z = ivy.less(x, y)
    >>> print(z)
    {
        a: ivy.array([[False, True, True],
                      [False, True, True],
                      [True, True, True]]),
        b: ivy.array([[False, True, True],
                      [True, True, True],
                      [True, True, True]])
    }

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([1, 2, 3]),b=ivy.array([5, 6, 7]))
    >>> z = ivy.less(x, y)
    >>> print(z)
    {
        a: ivy.array([False, False, False]),
        b: ivy.array([True, True, True])
    }

    """
    return ivy.current_backend(x1).less(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def log(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation to the natural (base ``e``)
    logarithm, having domain ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``,
    for each element ``x_i`` of the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

    - If ``a`` is ``-0`` and ``b`` is ``+0``, the result is ``-infinity + πj``.
    - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``-infinity + 0j``.
    - If ``a`` is a finite number and ``b`` is ``+infinity``,
      the result is ``+infinity + πj/2``.
    - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
    - If ``a`` is ``-infinity`` and ``b`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``+infinity + πj``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``+infinity + 0j``.
    - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``,
      the result is ``+infinity + 3πj/4``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``+infinity + πj/4``.
    - If ``a`` is either ``+infinity`` or ``-infinity`` and
      ``b`` is ``NaN``, the result is ``+infinity + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is a finite number,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``+infinity``,
      the result is ``+infinity + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.



    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the evaluated natural logarithm for each element in ``x``.
        The returned array must have a floating-point data type determined by
        :ref:`type-promotion`.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([4.0, 1, -0.0, -5.0])
    >>> y = ivy.log(x)
    >>> print(y)
    ivy.array([1.39, 0., -inf, nan])

    >>> x = ivy.array([[float('nan'), 1, 5.0, float('+inf')],
    ...                [+0, -1.0, -5, float('-inf')]])
    >>> y = ivy.log(x)
    >>> print(y)
    ivy.array([[nan, 0., 1.61, inf],
               [-inf, nan, nan, nan]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.0, float('nan')]),
    ...                   b=ivy.array([-0., -3.9, float('+inf')]),
    ...                   c=ivy.array([7.9, 1.1, 1.]))
    >>> y = ivy.log(x)
    >>> print(y)
    {
        a: ivy.array([-inf, nan]),
        b: ivy.array([-inf, nan, inf]),
        c: ivy.array([2.07, 0.0953, 0.])
    }
    """
    return ivy.current_backend(x).log(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def log10(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Calculate an implementation-dependent approximation to the base ``10``
    logarithm, having domain ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``,
    for each element ``x_i`` of the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    For complex floating-point operands, special cases must be handled as if the
    operation is implemented using the standard change of base formula

    .. math::
        \log_{10} x = \frac{\log_{e} x}{\log_{e} 10}

    where :math:`\log_{e}` is the natural logarithm.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the evaluated base ``10`` logarithm for each element in
        ``x``. The returned array must have a floating-point data type determined by
        :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.log10.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([4.0, 1, -0.0, -5.0])
    >>> y = ivy.log10(x)
    >>> print(y)
    ivy.array([0.602, 0., -inf, nan])

    >>> x = ivy.array([[float('nan'), 1, 5.0, float('+inf')],
    ...                [+0, -1.0, -5, float('-inf')]])
    >>> y = ivy.log10(x)
    >>> print(y)
    ivy.array([[nan, 0., 0.699, inf],
               [-inf, nan, nan, nan]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.0, float('nan')]),
    ...                   b=ivy.array([-0., -3.9, float('+inf')]),
    ...                   c=ivy.array([7.9, 1.1, 1.]))
    >>> y = ivy.log10(x)
    >>> print(y)
    {
        a: ivy.array([-inf, nan]),
        b: ivy.array([-inf, nan, inf]),
        c: ivy.array([0.898, 0.0414, 0.])
    }
    """
    return ivy.current_backend(x).log10(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def log1p(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate an implementation-dependent approximation to log(1+x), where log refers to
    the natural (base e) logarithm.

    .. note::
       The purpose of this function is to calculate ``log(1+x)`` more accurately
       when `x` is close to zero. Accordingly, conforming implementations should avoid
       implementing this function as simply ``log(1+x)``.
       See FDLIBM, or some other IEEE 754-2019 compliant mathematical library,
       for a potential reference implementation.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``-1``, the result is ``NaN``.
    - If ``x_i`` is ``-1``, the result is ``-infinity``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

    - If ``a`` is ``-1`` and ``b`` is ``+0``, the result is ``-infinity + 0j``.
    - If ``a`` is a finite number and ``b`` is ``+infinity``,
      the result is ``+infinity + πj/2``.
    - If ``a`` is a finite number and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``-infinity`` and ``b`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``+infinity + πj``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``+infinity + 0j``.
    - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``,
      the result is ``+infinity + 3πj/4``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``+infinity + πj/4``.
    - If ``a`` is either ``+infinity`` or ``-infinity`` and
      ``b`` is ``NaN``, the result is ``+infinity + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is a finite number,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``+infinity``,
      the result is ``+infinity + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.


    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the evaluated Natural logarithm of 1 + x for each element in
        ``x``. The returned array must have a floating-point data type determined by
        :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.log1p.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1 , 2 ,3 ])
    >>> y = ivy.log1p(x)
    >>> print(y)
    ivy.array([0.693, 1.1  , 1.39 ])

    >>> x = ivy.array([0 , 1 ])
    >>> y = ivy.zeros(2)
    >>> ivy.log1p(x , out = y)
    >>> print(y)
    ivy.array([0.   , 0.693])

    >>> x = ivy.array([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> ivy.log1p(x, out = x)
    >>> print(x)
    ivy.array([[0.742, 1.16 , 1.46 ],[1.69 , 1.87 , 2.03 ]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.1]))
    >>> y = ivy.log1p(x)
    >>> print(y)
    {
        a: ivy.array([0., 0.693, 1.1]),
        b: ivy.array([1.39, 1.61, 1.81])
    }
    """
    return ivy.current_backend(x).log1p(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def log2(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Calculate an implementation-dependent approximation to the base ``2`` logarithm,
    having domain ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``, for each
    element ``x_i`` of the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    For complex floating-point operands, special cases must be hanled as if
    the operation is implemented using the standard change of base formula

    .. math::
        \log_{2} x = \frac{\log_{e} x}{\log_{e} 2}

    where :math:`\log_{e}` is the natural logarithm.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the evaluated base ``2`` logarithm for each element in
        ``x``. The returned array must have a floating-point data type determined by
        :ref:`type-promotion`.


    This method conforms to the
    `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.log2.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return ivy.current_backend(x).log2(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def logaddexp(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate the logarithm of the sum of exponentiations ``log(exp(x1) + exp(x2))``
    for each element ``x1_i`` of the input array ``x1`` with the respective element
    ``x2_i`` of the input array ``x2``.

    **Special cases**

    For floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is not ``NaN``, the result is
      ``+infinity``.
    - If ``x1_i`` is not ``NaN`` and ``x2_i`` is ``+infinity``, the result is
      ``+infinity``.

    Parameters
    ----------
    x1
        first input array. Should have a floating-point data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.logaddexp.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([2., 5., 15.])
    >>> y = ivy.array([3., 2., 4.])
    >>> z = ivy.logaddexp(x, y)
    >>> print(z)
    ivy.array([ 3.31,  5.05, 15.  ])

    >>> x = ivy.array([[[1.1], [3.2], [-6.3]]])
    >>> y = ivy.array([[8.4], [2.5], [1.6]])
    >>> ivy.logaddexp(x, y, out=x)
    >>> print(x)
    ivy.array([[[8.4], [3.6], [1.6]]])

    With one :class:`ivy.Container` input:

    >>> x = ivy.array([[5.1, 2.3, -3.6]])
    >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),
    ...                   b=ivy.array([[5.], [6.], [7.]]))
    >>> z = ivy.logaddexp(x, y)
    >>> print(z)
    {
    a: ivy.array([[5.39, 4.17, 4.],
                  [5.74, 5.07, 5.],
                  [6.34, 6.02, 6.]]),
    b: ivy.array([[5.74, 5.07, 5.],
                  [6.34, 6.02, 6.],
                  [7.14, 7.01, 7.]])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([4., 5., 6.]),b=ivy.array([2., 3., 4.]))
    >>> y = ivy.Container(a=ivy.array([1., 2., 3.]),b=ivy.array([5., 6., 7.]))
    >>> z = ivy.logaddexp(y,x)
    >>> print(z)
    {
        a: ivy.array([4.05, 5.05, 6.05]),
        b: ivy.array([5.05, 6.05, 7.05])
    }

    """
    return ivy.current_backend(x1, x2).logaddexp(x1, x2, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def logaddexp2(
    x1: Union[ivy.Array, ivy.NativeArray, float, list, tuple],
    x2: Union[ivy.Array, ivy.NativeArray, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate log2(2**x1 + 2**x2).

    Parameters
    ----------
    x1
        First array-like input.
    x2
        Second array-input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Element-wise logaddexp2 of x1 and x2.

    Examples
    --------
    >>> x1 = ivy.array([1, 2, 3])
    >>> x2 = ivy.array([4, 5, 6])
    >>> ivy.logaddexp2(x1, x2)
    ivy.array([4.169925, 5.169925, 6.169925])
    """
    return ivy.current_backend(x1, x2).logaddexp2(x1, x2, out=out)


# ToDo: compare the examples against special case for zeros.


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def logical_and(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the logical AND for each element x1_i of the input array x1 with the
    respective element x2_i of the input array x2.

    Parameters
    ----------
    x1
        first input array. Should have a boolean data type.
    x2
        second input array. Must be compatible with x1.
        Should have a boolean data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type of bool.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.logical_and.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([True, True, False])
    >>> y = ivy.array([True, False, True])
    >>> print(ivy.logical_and(x, y))
    ivy.array([True,False,False])

    >>> ivy.logical_and(x, y, out=y)
    >>> print(y)
    ivy.array([True,False,False])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([False, True, True]),
    ...                   b=ivy.array([True, False, False]))
    >>> y = ivy.Container(a=ivy.array([True, True, False]),
    ...                   b=ivy.array([False, False, True]))
    >>> print(ivy.logical_and(y, x))
    {
        a: ivy.array([False, True, False]),
        b: ivy.array([False, False, False])
    }

    >>> ivy.logical_and(y, x, out=y)
    >>> print(y)
    {
        a: ivy.array([False, True, False]),
        b: ivy.array([False, False, False])
    }


    >>> x = ivy.Container(a=ivy.array([False, True, True]),
    ...                   b=ivy.array([True, False, False]))
    >>> y = ivy.array([True, False, True])
    >>> print(ivy.logical_and(y, x))
    {
        a: ivy.array([False, False, True]),
        b: ivy.array([True, False, False])
    }

    >>> x = ivy.Container(a=ivy.array([False, True, True]),
    ...                   b=ivy.array([True, False, False]))
    >>> y = ivy.array([True, False, True])
    >>> ivy.logical_and(y, x, out=x)
    >>> print(x)
    {
        a: ivy.array([False, False, True]),
        b: ivy.array([True, False, False])
    }
    """
    return ivy.current_backend(x1, x2).logical_and(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def logical_not(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the logical NOT for each element ``x_i`` of the input array ``x``.

    .. note::
       While this specification recommends that this function only accept input arrays
       having a boolean data type, specification-compliant array libraries may choose to
       accept input arrays having numeric data types. If non-boolean data types are
       supported, zeros must be considered the equivalent of ``False``, while non-zeros
       must be considered the equivalent of ``True``.

       **Special cases**

       For this particular case,

       - If ``x_i`` is ``NaN``, the result is ``False``.
       - If ``x_i`` is ``-0``, the result is ``True``.
       - If ``x_i`` is ``-infinity``, the result is ``False``.
       - If ``x_i`` is ``+infinity``, the result is ``False``.

    Parameters
    ----------
    x
        input array. Should have a boolean data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type of ``bool``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.logical_not.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x=ivy.array([1,0,1,1,0])
    >>> y=ivy.logical_not(x)
    >>> print(y)
    ivy.array([False, True, False, False,  True])

    >>> x=ivy.array([2,0,3,5])
    >>> y=ivy.logical_not(x)
    >>> print(y)
    ivy.array([False, True, False, False])

    >>> x=ivy.native_array([1,0,6,5])
    >>> y=ivy.logical_not(x)
    >>> print(y)
    ivy.array([False, True, False, False])

    With :class:`ivy.Container` input:

    >>> x=ivy.Container(a=ivy.array([1,0,1,1]), b=ivy.array([1,0,8,9]))
    >>> y=ivy.logical_not(x)
    >>> print(y)
    {
        a: ivy.array([False, True, False, False]),
        b: ivy.array([False, True, False, False])
    }

    >>> x=ivy.Container(a=ivy.array([1,0,1,0]), b=ivy.native_array([5,2,0,3]))
    >>> y=ivy.logical_not(x)
    >>> print(y)
    {
        a: ivy.array([False, True, False, True]),
        b: ivy.array([False, False, True, False])
    }
    """
    return ivy.current_backend(x).logical_not(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def logical_or(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the logical OR for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    .. note::
       While this specification recommends that this function only accept input arrays
       having a boolean data type, specification-compliant array libraries may choose to
       accept input arrays having numeric data types. If non-boolean data types are
       supported, zeros must be considered the equivalent of ``False``, while non-zeros
       must be considered the equivalent of ``True``.

    Parameters
    ----------
    x1
        first input array. Should have a boolean data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have a boolean data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type of ``bool``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.logical_or.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([True, False, True])
    >>> y = ivy.array([True, True, False])
    >>> print(ivy.logical_or(x, y))
    ivy.array([ True,  True,  True])

    >>> x = ivy.array([[False, False, True], [True, False, True]])
    >>> y = ivy.array([[False, True, False], [True, True, False]])
    >>> z = ivy.zeros_like(x)
    >>> ivy.logical_or(x, y, out=z)
    >>> print(z)
    ivy.array([[False,  True,  True],
       [ True,  True,  True]])

    >>> x = ivy.array([False, 3, 0])
    >>> y = ivy.array([2, True, False])
    >>> ivy.logical_or(x, y, out=x)
    >>> print(x)
    ivy.array([ True,  True, False])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([False, False, True]),
    ...                   b=ivy.array([True, False, True]))
    >>> y = ivy.Container(a=ivy.array([False, True, False]),
    ...                   b=ivy.array([True, True, False]))
    >>> z = ivy.logical_or(x, y)
    >>> print(z)
    {
        a: ivy.array([False, True, True]),
        b: ivy.array([True, True, True])
    }
    """
    return ivy.current_backend(x1, x2).logical_or(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def logical_xor(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the bitwise XOR of the underlying binary representation of each element
    ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have an integer or boolean data type.
    x2
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
        Should have an integer or boolean data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.logical_xor.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type determined by :ref:`type-promotion`.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([1,0,1,1,0])
    >>> y = ivy.array([1,0,1,1,0])
    >>> z = ivy.logical_xor(x,y)
    >>> print(z)
    ivy.array([False, False, False, False, False])

    >>> x = ivy.array([[[1], [2], [3], [4]]])
    >>> y = ivy.array([[[4], [5], [6], [7]]])
    >>> z = ivy.logical_xor(x,y)
    >>> print(z)
    ivy.array([[[False],
            [False],
            [False],
            [False]]])

    >>> x = ivy.array([[[1], [2], [3], [4]]])
    >>> y = ivy.array([4, 5, 6, 7])
    >>> z = ivy.logical_xor(x,y)
    >>> print(z)
    ivy.array([[[False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False]]])

    With :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([1,0,0,1,0]), b=ivy.array([1,0,1,0,0]))
    >>> y = ivy.Container(a=ivy.array([0,0,1,1,0]), b=ivy.array([1,0,1,1,0]))
    >>> z = ivy.logical_xor(x,y)
    >>> print(z)
    {
    a: ivy.array([True, False, True, False, False]),
    b: ivy.array([False, False, False, True, False])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([1,0,0,1,0]), b=ivy.array([1,0,1,0,0]))
    >>> y = ivy.array([0,0,1,1,0])
    >>> z = ivy.logical_xor(x,y)
    >>> print(z)
    {
    a: ivy.array([True, False, True, False, False]),
    b: ivy.array([True, False, False, True, False])
    }
    """
    return ivy.current_backend(x1, x2).logical_xor(x1, x2, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def nan_to_num(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: bool = True,
    nan: Union[float, int] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Replace NaN with zero and infinity with large finite numbers (default behaviour) or
    with the numbers defined by the user using the nan, posinf and/or neginf keywords.

    Parameters
    ----------
    x
        Array input.
    copy
        Whether to create a copy of x (True) or to replace values in-place (False).
        The in-place operation only occurs if casting to an array does not require
        a copy. Default is True.
    nan
        Value to be used to fill NaN values. If no value is passed then NaN values
        will be replaced with 0.0.
    posinf
        Value to be used to fill positive infinity values. If no value is passed
        then positive infinity values will be replaced with a very large number.
    neginf
        Value to be used to fill negative infinity values.
        If no value is passed then negative infinity values
        will be replaced with a very small (or negative) number.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with the non-finite values replaced.
        If copy is False, this may be x itself.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3, nan])
    >>> ivy.nan_to_num(x)
    ivy.array([1.,    1.,   3.,   0.0])
    >>> x = ivy.array([1, 2, 3, inf])
    >>> ivy.nan_to_num(x, posinf=5e+100)
    ivy.array([1.,   2.,   3.,   5e+100])
    """
    return ivy.current_backend(x).nan_to_num(
        x, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def negative(
    x: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array with the negative value of each element in ``x``.

    .. note::
       For signed integer data types, the numerical negative of
       the minimum representable integer is implementation-dependent.

    .. note::
       If ``x`` has a complex floating-point data type,
       both the real and imaginary components for each ``x_i``
       must be negated (a result which follows from the rules of
       complex number multiplication).

    Parameters
    ----------
    x
        Input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        A new array with the negative value of each element in ``x``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.negative.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0,1,1,2])
    >>> y = ivy.negative(x)
    >>> print(y)
    ivy.array([ 0, -1, -1, -2])

    >>> x = ivy.array([0,-1,-0.5,2,3])
    >>> y = ivy.zeros(5)
    >>> ivy.negative(x, out=y)
    >>> print(y)
    ivy.array([-0. ,  1. ,  0.5, -2. , -3. ])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.negative(x,out=x)
    >>> print(x)
    ivy.array([[-1.1, -2.2, -3.3],
       [4.4, 5.5, 6.6]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., -5.]))
    >>> y = ivy.negative(x)
    >>> print(y)
    {
        a: ivy.array([-0., -1., -2.]),
        b: ivy.array([-3., -4., 5.])
    }

    """
    return ivy.current_backend(x).negative(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def not_equal(
    x1: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
    x2: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the truth value of ``x1_i != x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    **Special Cases**

    For real-valued floating-point operands,

    - If ``x1_i`` is ``NaN`` or ``x2_i`` is ``NaN``, the result is ``True``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``-infinity``,
      the result is ``True``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``+infinity``,
      the result is ``True``.
    - If ``x1_i`` is a finite number, ``x2_i`` is a finite number,
      and ``x1_i`` does not equal ``x2_i``, the result is ``True``.
    - In the remaining cases, the result is ``False``.

    For omplex floating-point operands, let ``a = real(x1_i)``, ``b = imag(x1_i)``,
    ``c = real(x2_i)``, ``d = imag(x2_i)``, and

    - If ``a``, ``b``, ``c``, or ``d`` is ``NaN``, the result is ``True``.
    - In the remaining cases, the result is the logical OR of
      the equality comparison between the real values ``a`` and ``c``
      (real components) and between the real values ``b`` and ``d``
      (imaginary components), as described above for real-valued floating-point operands
      (i.e., ``a != c OR b != d``).

    Parameters
    ----------
    x1
        first input array. Should have a numeric data type.
    x2
        second input array. Must be compatible with ``x1`` (see  ref:`broadcasting`).
        Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type of ``bool``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.not_equal.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    ------------------

    With :class:`ivy.Array` inputs:

    >>> x1 = ivy.array([1, 0, 1, 1])
    >>> x2 = ivy.array([1, 0, 0, -1])
    >>> y = ivy.not_equal(x1, x2)
    >>> print(y)
    ivy.array([False, False, True, True])

    >>> x1 = ivy.array([1, 0, 1, 0])
    >>> x2 = ivy.array([0, 1, 0, 1])
    >>> y = ivy.not_equal(x1, x2)
    >>> print(y)
    ivy.array([True, True, True, True])

    >>> x1 = ivy.array([1, -1, 1, -1])
    >>> x2 = ivy.array([0, -1, 1, 0])
    >>> y = ivy.zeros(4)
    >>> ivy.not_equal(x1, x2, out=y)
    >>> print(y)
    ivy.array([True, False, False, True])

    >>> x1 = ivy.array([1, -1, 1, -1])
    >>> x2 = ivy.array([0, -1, 1, 0])
    >>> y = ivy.not_equal(x1, x2, out=x1)
    >>> print(y)
    ivy.array([True, False, False, True])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x1 = ivy.native_array([1, 2])
    >>> x2 = ivy.array([1, 2])
    >>> y = ivy.not_equal(x1, x2)
    >>> print(y)
    ivy.array([False, False])

    >>> x1 = ivy.native_array([1, -1])
    >>> x2 = ivy.array([0, 1])
    >>> y = ivy.not_equal(x1, x2)
    >>> print(y)
    ivy.array([True, True])

    >>> x1 = ivy.native_array([1, -1, 1, -1])
    >>> x2 = ivy.native_array([0, -1, 1, 0])
    >>> y = ivy.zeros(4)
    >>> ivy.not_equal(x1, x2, out=y)
    >>> print(y)
    ivy.array([True, False, False, True])

    >>> x1 = ivy.native_array([1, 2, 3, 4])
    >>> x2 = ivy.native_array([0, 2, 3, 4])
    >>> y = ivy.zeros(4)
    >>> ivy.not_equal(x1, x2, out=y)
    >>> print(y)
    ivy.array([True, False, False, False])

    With :class:`ivy.Container` input:

    >>> x1 = ivy.Container(a=ivy.array([1, 0, 3]),
    ...                    b=ivy.array([1, 2, 3]),
    ...                    c=ivy.native_array([1, 2, 4]))
    >>> x2 = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                    b=ivy.array([1, 2, 3]),
    ...                    c=ivy.native_array([1, 2, 4]))
    >>> y = ivy.not_equal(x1, x2)
    >>> print(y)
    {
        a: ivy.array([False, True, False]),
        b: ivy.array([False, False, False]),
        c: ivy.array([False, False, False])
    }

    >>> x1 = ivy.Container(a=ivy.native_array([0, 1, 0]),
    ...                    b=ivy.array([1, 2, 3]),
    ...                    c=ivy.native_array([1.0, 2.0, 4.0]))
    >>> x2 = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                    b=ivy.native_array([1.1, 2.1, 3.1]),
    ...                    c=ivy.native_array([1, 2, 4]))
    >>> y = ivy.not_equal(x1, x2)
    >>> print(y)
    {
        a: ivy.array([True, True, True]),
        b: ivy.array([True, True, True]),
        c: ivy.array([False, False, False])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                    b=ivy.array([1, 3, 5]))
    >>> x2 = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                    b=ivy.array([1, 4, 5]))
    >>> y = ivy.not_equal(x1, x2)
    >>> print(y)
    {
        a: ivy.array([False, False, False]),
        b: ivy.array([False, True, False])
    }

    >>> x1 = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]),
    ...                    b=ivy.array([1, 4, 5]))
    >>> x2 = ivy.Container(a=ivy.array([1, 2, 3.0]),
    ...                    b=ivy.array([1.0, 4.0, 5.0]))
    >>> y = ivy.not_equal(x1, x2)
    >>> print(y)
    {
        a: ivy.array([False, False, False]),
        b: ivy.array([False, False, False])
    }
    """
    return ivy.current_backend(x1, x2).not_equal(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def positive(
    x: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array with the positive value of each element in ``x``.

    Parameters
    ----------
    x
        Input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        A new array with the positive value of each element in ``x``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.positive.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([2, 3 ,5, 7])
    >>> y = ivy.positive(x)
    >>> print(y)
    ivy.array([2, 3, 5, 7])

    >>> x = ivy.array([0, -1, -0.5, 2, 3])
    >>> y = ivy.zeros(5)
    >>> ivy.positive(x, out=y)
    >>> print(y)
    ivy.array([0., -1., -0.5,  2.,  3.])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.positive(x,out=x)
    >>> print(x)
    ivy.array([[ 1.1,  2.2,  3.3],
       [-4.4, -5.5, -6.6]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., -5.]))
    >>> y = ivy.positive(x)
    >>> print(y)
    {
    a: ivy.array([0., 1., 2.]),
    b: ivy.array([3., 4., -5.])
    }

    """
    return ivy.current_backend(x).positive(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def pow(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate an implementation-dependent approximation of exponentiation by raising
    each element ``x1_i`` (the base) of the input array ``x1`` to the power of ``x2_i``
    (the exponent), where ``x2_i`` is the corresponding element of the input array
    ``x2``.

    .. note::
       If both ``x1`` and ``x2`` have integer data types, the result of ``pow`` when
       ``x2_i`` is negative (i.e., less than zero) is unspecified and thus
       implementation-dependent. If ``x1`` has an integer data type and ``x2`` has a
       floating-point data type, behavior is implementation-dependent (type promotion
       between data type "kinds" (integer versus floating-point) is unspecified).

    **Special cases**

    For floating-point operands,

    - If ``x1_i`` is not equal to ``1`` and ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x2_i`` is ``+0``, the result is ``1``, even if ``x1_i`` is ``NaN``.
    - If ``x2_i`` is ``-0``, the result is ``1``, even if ``x1_i`` is ``NaN``.
    - If ``x1_i`` is ``NaN`` and ``x2_i`` is not equal to ``0``, the result is ``NaN``.
    - If ``abs(x1_i)`` is greater than ``1`` and ``x2_i`` is ``+infinity``, the result
      is ``+infinity``.
    - If ``abs(x1_i)`` is greater than ``1`` and ``x2_i`` is ``-infinity``, the result
      is ``+0``.
    - If ``abs(x1_i)`` is ``1`` and ``x2_i`` is ``+infinity``, the result is ``1``.
    - If ``abs(x1_i)`` is ``1`` and ``x2_i`` is ``-infinity``, the result is ``1``.
    - If ``x1_i`` is ``1`` and ``x2_i`` is not ``NaN``, the result is ``1``.
    - If ``abs(x1_i)`` is less than ``1`` and ``x2_i`` is ``+infinity``, the result is
      ``+0``.
    - If ``abs(x1_i)`` is less than ``1`` and ``x2_i`` is ``-infinity``, the result is
      ``+infinity``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is greater than ``0``, the result is
      ``+infinity``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is less than ``0``, the result is
      ``+0``.
    - If ``x1_i`` is ``-infinity``, ``x2_i`` is greater than ``0``, and ``x2_i`` is an
      odd integer value, the result is ``-infinity``.
    - If ``x1_i`` is ``-infinity``, ``x2_i`` is greater than ``0``, and ``x2_i`` is not
      an odd integer value, the result is ``+infinity``.
    - If ``x1_i`` is ``-infinity``, ``x2_i`` is less than ``0``, and ``x2_i`` is an odd
      integer value, the result is ``-0``.
    - If ``x1_i`` is ``-infinity``, ``x2_i`` is less than ``0``, and ``x2_i`` is not an
      odd integer value, the result is ``+0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is
      ``+infinity``.
    - If ``x1_i`` is ``-0``, ``x2_i`` is greater than ``0``, and ``x2_i`` is an odd
      integer value, the result is ``-0``.
    - If ``x1_i`` is ``-0``, ``x2_i`` is greater than ``0``, and ``x2_i`` is not an odd
      integer value, the result is ``+0``.
    - If ``x1_i`` is ``-0``, ``x2_i`` is less than ``0``, and ``x2_i`` is an odd integer
      value, the result is ``-infinity``.
    - If ``x1_i`` is ``-0``, ``x2_i`` is less than ``0``, and ``x2_i`` is not an odd
      integer value, the result is ``+infinity``.
    - If ``x1_i`` is less than ``0``, ``x1_i`` is a finite number, ``x2_i`` is a finite
      number, and ``x2_i`` is not an integer value, the result is ``NaN``.

    For complex floating-point operands, special cases should be handled
    as if the operation is implemented as ``exp(x2*log(x1))``.

    .. note::
       Conforming implementations are allowed to treat special cases involving
       complex floating-point operands more carefully than as described
       in this specification.

    Parameters
    ----------
    x1
        first input array whose elements correspond to the exponentiation base. Should
        have a numeric data type.
    x2
        second input array whose elements correspond to the exponentiation exponent.
        Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric
        data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.pow.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.pow(x, 3)
    >>> print(y)
    ivy.array([1, 8, 27])

    >>> x = ivy.array([1.5, -0.8, 0.3])
    >>> y = ivy.zeros(3)
    >>> ivy.pow(x, 2, out=y)
    >>> print(y)
    ivy.array([2.25, 0.64, 0.09])

    >>> x = ivy.array([[1.2, 2, 3.1], [1, 2.5, 9]])
    >>> ivy.pow(x, 2.3, out=x)
    >>> print(x)
    ivy.array([[  1.52095687,   4.92457771,  13.49372482],
           [  1.        ,   8.22738838, 156.5877228 ]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0, 1]), b=ivy.array([2, 3]))
    >>> y = ivy.pow(x, 3)
    >>> print(y)
    {
        a:ivy.array([0,1]),
        b:ivy.array([8,27])
    }
    """
    return ivy.current_backend(x1, x2).pow(x1, x2, out=out)


pow.unsupported_gradients = {"torch": ["float16"]}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def real(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Test each element ``x_i`` of the input array ``x`` to
    take only real part from it.
    Returns a float array, where it only contains .
    If element has complex type with zero complex part, the return value
    will be that element, else it only returns real part.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing test results. An element ``out_i`` is
        ``real number`` if ``x_i`` contain real number part only
        and if it is ``real number with complex part also`` then it
        returns the real number part.
        The returned array must have a floating-point data type with the
        same floating-point precision as ``x`` (e.g., if ``x`` is ``complex64``,
        the returned array must have the floating-point precision of ``float32``).

    The descriptions above assume an array input for simplicity, but
    the method also accepts :class:`ivy.Container` instances
    in place of: class:`ivy.Array` or :class:`ivy.NativeArray`
    instances, as shown in the type hints and also the examples below.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([[[1.1], [2], [-6.3]]])
    >>> z = ivy.real(x)
    >>> print(z)
    ivy.array([[[1.1], [2.], [-6.3]]])

    >>> x = ivy.array([4.2-0j, 3j, 7+5j])
    >>> z = ivy.real(x)
    >>> print(z)
    ivy.array([4.2, 0., 7.])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-6.7-7j, 0.314+0.355j, 1.23]),\
                          b=ivy.array([5j, 5.32-6.55j, 3.001]))
    >>> z = ivy.real(x)
    >>> print(z)
    {
        a: ivy.array([-6.7, 0.314, 1.23]),
        b: ivy.array([0., 5.32, 3.001])
    }
    """
    return ivy.current_backend(x).real(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def remainder(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    modulus: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the remainder of division for each element ``x1_i`` of the input array
    ``x1`` and the respective element ``x2_i`` of the input array ``x2``.

    .. note::
        This function is equivalent to the Python modulus operator ``x1_i % x2_i``. For
        input arrays which promote to an integer data type, the result of division by
        zero is unspecified and thus implementation-defined. In general, similar to
        Python’s ``%`` operator, this function is not recommended for floating-point
        operands as semantics do not follow IEEE 754. That this function is specified
        to accept floating-point operands is primarily for reasons of backward
        compatibility.

    **Special Cases**

    For floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either
      ``+infinity`` or ``-infinity``, the result is ``NaN``.
    - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``,
      the result is ``NaN``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is ``NaN``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is ``NaN``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is ``NaN``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is ``NaN``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``)
      finite number, the result is ``NaN``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative (i.e., less than ``0``)
      finite number, the result is ``NaN``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``)
      finite number, the result is ``NaN``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative (i.e., less than ``0``)
      finite number, the result is ``NaN``.
    - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is
      ``+infinity``, the result is ``x1_i``. (note: this result matches Python
      behavior.)
    - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is
      ``-infinity``, the result is ``x2_i``. (note: this result matches Python
      behavior.)
    - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is
      ``+infinity``, the result is ``x2_i``. (note: this results matches Python
      behavior.)
    - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is
      ``-infinity``, the result is ``x1_i``. (note: this result matches Python
      behavior.)
    - In the remaining cases, the result must match that of the Python ``%`` operator.

    Parameters
    ----------
    x1
        dividend input array. Should have a numeric data type.
    x2
        divisor input array. Must be compatible with ``x1`` (see  ref:`Broadcasting`).
        Should have a numeric data type.
    modulus
        whether to compute the modulus instead of the remainder. Default is ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. Each element-wise result must have
        the same sign as the respective element ``x2_i``. The returned array must have a
        data type determined by :ref:`Type Promotion Rules`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.remainder.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x1 = ivy.array([2., 5., 15.])
    >>> x2 = ivy.array([3., 2., 4.])
    >>> y = ivy.remainder(x1, x2)
    >>> print(y)
    ivy.array([2., 1., 3.])

    With mixed :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x1 = ivy.array([23., 1., 6.])
    >>> x2 = ivy.native_array([11., 2., 4.])
    >>> y = ivy.remainder(x1, x2)
    >>> print(y)
    ivy.array([1., 1., 2.])

    With :class:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([2., 3., 5.]), b=ivy.array([2., 2., 4.]))
    >>> x2 = ivy.Container(a=ivy.array([1., 3., 4.]), b=ivy.array([1., 3., 3.]))
    >>> y = ivy.remainder(x1, x2)
    >>> print(y)
    {
        a: ivy.array([0., 0., 1.]),
        b: ivy.array([0., 2., 1.])
    }
    """
    return ivy.current_backend(x1, x2).remainder(x1, x2, modulus=modulus, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def round(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    decimals: Optional[int] = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Round each element ``x_i`` of the input array ``x`` to the nearest
    integer-valued number.

    .. note::
       For complex floating-point operands, real and imaginary components
       must be independently rounded to the nearest integer-valued number.

       Rounded real and imaginary components must be equal
       to their equivalent rounded real-valued floating-point
       counterparts (i.e., for complex-valued ``x``, ``real(round(x))``
       must equal ``round(real(x)))`` and ``imag(round(x))`` must equal
       ``round(imag(x))``).

    **Special cases**

    - If ``x_i`` is already an integer-valued, the result is ``x_i``.

    For floating-point operands,

    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If two integers are equally close to ``x_i``, the result is
      the even integer closest to ``x_i``.

    .. note::
       For complex floating-point operands, the following special
       cases apply to real and imaginary components independently
       (e.g., if ``real(x_i)`` is ``NaN``, the rounded
       real component is ``NaN``).

    - If ``x_i`` is already integer-valued, the result is ``x_i``.

    Parameters
    ----------
    x
        input array containing elements to round.
    decimals
        number of decimal places to round to. Default is ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array of the same shape and type as x, with the elements rounded to integers.


    Note: PyTorch supports an additional argument :code:`decimals` for the
    `round function <https://pytorch.org/docs/stable/generated/torch.round.html>`_.
    It has been deliberately omitted here due to the imprecise
    nature of the argument in :code:`torch.round`.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.round.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1.2, 2.4, 3.6])
    >>> y = ivy.round(x)
    >>> print(y)
    ivy.array([1.,2.,4.])

    >>> x = ivy.array([-0, 5, 4.5])
    >>> y = ivy.round(x)
    >>> print(y)
    ivy.array([0.,5.,4.])

    >>> x = ivy.array([1.5654, 2.034, 15.1, -5.0])
    >>> y = ivy.zeros(4)
    >>> ivy.round(x, out=y)
    >>> print(y)
    ivy.array([2.,2.,15.,-5.])

    >>> x = ivy.array([[0, 5.433, -343.3, 1.5],
    ...                [-5.5, 44.2, 11.5, 12.01]])
    >>> ivy.round(x, out=x)
    >>> print(x)
    ivy.array([[0.,5.,-343.,2.],[-6.,44.,12.,12.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([4.20, 8.6, 6.90, 0.0]),
    ...                   b=ivy.array([-300.9, -527.3, 4.5]))
    >>> y = ivy.round(x)
    >>> print(y)
    {
        a:ivy.array([4.,9.,7.,0.]),
        b:ivy.array([-301.,-527.,4.])
    }
    """
    return ivy.current_backend(x).round(x, decimals=decimals, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def sign(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    np_variant: Optional[bool] = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Return an indication of the sign of a number for each element ``x_i`` of the
    input array ``x``.

    The sign function (also known as the **signum function**)
    of a number :math:`x_{i}` is defined as

    .. math::
        \operatorname{sign}(x_i) = \begin{cases}
       0 & \textrm{if } x_i = 0 \\
       \frac{x}{|x|} & \textrm{otherwise}
       \end{cases}

    where :math:`|x_i|` is the absolute value of :math:`x_i`.

    **Special cases**

    - If ``x_i`` is less than ``0``, the result is ``-1``.
    - If ``x_i`` is either ``-0`` or ``+0``, the result is ``0``.
    - If ``x_i`` is greater than ``0``, the result is ``+1``.
    - For complex numbers ``sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j``

    For complex floating-point operands, let ``a = real(x_i)``,
    ``b = imag(x_i)``, and

    - If ``a`` is either ``-0`` or ``+0`` and ``b`` is
      either ``-0`` or ``+0``, the result is ``0 + 0j``.
    - If ``a`` is ``NaN`` or ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - In the remaining cases, special cases must be handled
      according to the rules of complex number division.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    np_variant
        Handles complex numbers like numpy does If ``True``,
        ``sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j``.
        otherwise, For complex numbers, ``y = sign(x) = x / |x| if x != 0,
        otherwise y = 0.``

    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the evaluated result for each element in ``x``. The returned
        array must have the same data type as ``x``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.sign.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([8.3, -0, 6.8, 0.07])
    >>> y = ivy.sign(x)
    >>> print(y)
    ivy.array([1., 0., 1., 1.])

    >>> x = ivy.array([[5.78, -4., -6.9, 0],
    ...                [-.4, 0.5, 8, -0.01]])
    >>> y = ivy.sign(x)
    >>> print(y)
    ivy.array([[ 1., -1., -1.,  0.],
               [-1.,  1.,  1., -1.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., -0.]),
    ...                   b=ivy.array([1.46, 5.9, -0.0]),
    ...                   c=ivy.array([-8.23, -4.9, -2.6, 7.4]))
    >>> y = ivy.sign(x)
    >>> print(y)
    {
        a: ivy.array([0., 0.]),
        b: ivy.array([1., 1., 0.]),
        c: ivy.array([-1., -1., -1., 1.])
    }
    """
    return ivy.current_backend(x).sign(x, np_variant=np_variant, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def sin(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Calculate an implementation-dependent approximation to the sine, having domain
    ``(-infinity, +infinity)`` and codomain ``[-1, +1]``, for each element ``x_i`` of
    the input array ``x``. Each element ``x_i`` is assumed to be expressed in radians.

    .. note::
       The sine is an entire function on the complex plane and has no branch cuts.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

    For complex floating-point operands, special cases
    must be handled as if the operation is implemented as ``-1j * sinh(x*1j)``.

    Parameters
    ----------
    x
        input array whose elements are each expressed in radians. Should have a
        floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the sine of each element in ``x``. The returned array must
        have a floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.sin.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.sin(x)
    >>> print(y)
    ivy.array([0., 0.841, 0.909])

    >>> x = ivy.array([0., 1.2, -2.3, 3.6])
    >>> y = ivy.zeros(4)
    >>> ivy.sin(x, out=y)
    >>> print(y)
    ivy.array([0., 0.932, -0.746, -0.443])

    >>> x = ivy.array([[1., 2., 3.], [-4., -5., -6.]])
    >>> ivy.sin(x, out=x)
    >>> print(x)
    ivy.array([[0.841, 0.909, 0.141],
               [0.757, 0.959, 0.279]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2., 3.]),
    ...                   b=ivy.array([-4., -5., -6., -7.]))
    >>> y = ivy.sin(x)
    >>> print(y)
    {
        a: ivy.array([0., 0.841, 0.909, 0.141]),
        b: ivy.array([0.757, 0.959, 0.279, -0.657])
    }
    """
    return ivy.current_backend(x).sin(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def sinh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Calculate an implementation-dependent approximation to the hyperbolic sine,
    having domain ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``,
    for each element ``x_i`` of the input array ``x``.

    .. math::
       \operatorname{sinh}(x) = \frac{e^x - e^{-x}}{2}

    .. note::
       The hyperbolic sine is an entire function in the
       complex plane and has no branch cuts.
       The function is periodic, with period
       :math:`2\pi j`, with respect to the imaginary component.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.

    For complex floating-point operands, let ``a = real(x_i)``,
    ``b = imag(x_i)``, and

    .. note::
       For complex floating-point operands, ``sinh(conj(x))``
       must equal ``conj(sinh(x))``.

    - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
    - If ``a`` is ``+0`` and ``b`` is ``+infinity``,
      the result is ``0 + NaN j`` (sign of the real component is unspecified).
    - If ``a`` is ``+0`` and ``b`` is ``NaN``,
      the result is ``0 + NaN j`` (sign of the real component is unspecified).
    - If ``a`` is a positive (i.e., greater than ``0``)
      finite number and ``b`` is ``+infinity``, the result is ``NaN + NaN j``.
    - If ``a`` is a positive (i.e., greater than ``0``)
      finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+0``,
      the result is ``+infinity + 0j``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive finite number,
      the result is ``+infinity * cis(b)``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``infinity + NaN j`` (sign of the real component is unspecified).
    - If ``a`` is ``+infinity`` and ``b`` is ``NaN``,
      the result is ``infinity + NaN j``
      (sign of the real component is unspecified).
    - If ``a`` is ``NaN`` and ``b`` is ``+0``, the result is ``NaN + 0j``.
    - If ``a`` is ``NaN`` and ``b`` is a nonzero finite number,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.

    where ``cis(v)`` is ``cos(v) + sin(v)*1j``.

    Parameters
    ----------
    x
        input array whose elements each represent a hyperbolic angle. Should have a
        floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the hyperbolic sine of each element in ``x``. The returned
        array must have a floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.sinh.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.sinh(x)
    >>> print(y)
        ivy.array([1.18, 3.63, 10.])

    >>> x = ivy.array([0.23, 3., -1.2])
    >>> ivy.sinh(x, out=x)
    >>> print(x)
        ivy.array([0.232, 10., -1.51])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.23, -0.25, 1]), b=ivy.array([3, -4, 1.26]))
    >>> y = ivy.sinh(x)
    >>> print(y)
    {
        a: ivy.array([0.232, -0.253, 1.18]),
        b: ivy.array([10., -27.3, 1.62])
    }
    """
    return ivy.current_backend(x).sinh(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def sqrt(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Calculate the square root, having domain ``[0, +infinity]`` and codomain
    ``[0, +infinity]``, for each element ``x_i`` of the input array ``x``. After
    rounding, each result must be indistinguishable from the infinitely precise result
    (as required by IEEE 754).

    .. note::
       After rounding, each result must be indistinguishable
       from the infinitely precise result (as required by IEEE 754).

    .. note::
       For complex floating-point operands, ``sqrt(conj(x))``
       must equal ``conj(sqrt(x))``.

    .. note::
       By convention, the branch cut of the square root is
       the negative real axis :math:`(-\infty, 0)`.

       The square root is a continuous function from above
       the branch cut, taking into account the sign of the imaginary component.

       Accordingly, for complex arguments, the function returns
       the square root in the range of the right half-plane,
       including the imaginary axis (i.e., the plane defined by
       :math:`[0, +\infty)` along the real axis and :math:`(-\infty, +\infty)`
       along the imaginary axis).

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    For complex floating-point operands, let ``a = real(x_i)``,
    ``b = imag(x_i)``, and

    - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``,
      the result is ``+0 + 0j``.
    - If ``a`` is any value (including ``NaN``) and ``b`` is
      ``+infinity``, the result is ``+infinity + infinity j``.
    - If ``a`` is a finite number and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.
    - If ``a`` ``-infinity`` and ``b`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``NaN + NaN j``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``+0 + infinity j``.
    - If ``a`` is ``-infinity`` and ``b`` is ``NaN``,
      the result is ``NaN + infinity j``
      (sign of the imaginary component is unspecified).
    - If ``a`` is ``+infinity`` and ``b`` is ``NaN``,
      the result is ``+infinity + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is any value,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.


    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the square root of each element in ``x``. The returned array
        must have a floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.sqrt.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, 4., 8.])
    >>> y = ivy.sqrt(x)
    >>> print(y)
    ivy.array([0., 2., 2.83])

    >>> x = ivy.array([1, 2., 4.])
    >>> y = ivy.zeros(3)
    >>> ivy.sqrt(x, out=y)
    ivy.array([1., 1.41, 2.])

    >>> X = ivy.array([40., 24., 100.])
    >>> ivy.sqrt(x, out=x)
    >>> ivy.array([6.32455532, 4.89897949, 10.])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([44., 56., 169.]), b=ivy.array([[49.,1.], [0,20.]])) # noqa
    >>> y = ivy.sqrt(x)
    >>> print(y)
    {
        a: ivy.array([6.63, 7.48, 13.]),
        b: ivy.array([[7., 1.],
                      [0., 4.47]])
    }
    """
    return ivy.current_backend(x).sqrt(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def square(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        Input array. Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the evaluated result for each element in ``x``.


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.square.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.square(x)
    >>> print(y)
    ivy.array([1, 4, 9])

    >>> x = ivy.array([1.5, -0.8, 0.3])
    >>> y = ivy.zeros(3)
    >>> ivy.square(x, out=y)
    >>> print(y)
    ivy.array([2.25, 0.64, 0.09])

    >>> x = ivy.array([[1.2, 2, 3.1], [-1, -2.5, -9]])
    >>> ivy.square(x, out=x)
    >>> print(x)
    ivy.array([[1.44,4.,9.61],[1.,6.25,81.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0, 1]), b=ivy.array([2, 3]))
    >>> y = ivy.square(x)
    >>> print(y)
    {
        a:ivy.array([0,1]),
        b:ivy.array([4,9])
    }
    """
    return ivy.current_backend(x).square(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def subtract(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate the difference for each element ``x1_i`` of the input array ``x1``
    with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a numeric data type.
    x2
        second input array. Must be compatible with ``x1`` (see  ref:`broadcasting`).
        Should have a numeric data type.
    alpha
        optional scalar multiplier for ``x2``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise differences.


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.subtract.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = ivy.array([3, 6, 3])
    >>> y = ivy.array([2, 1, 6])
    >>> z = ivy.subtract(x, y)
    >>> print(z)
    ivy.array([ 1,  5, -3])

    >>> x = ivy.array([3, 6, 3])
    >>> y = ivy.array([2, 1, 6])
    >>> z = ivy.subtract(x, y, alpha=2)
    >>> print(z)
    ivy.array([-1,  4, -9])

    """
    return ivy.current_backend(x1).subtract(x1, x2, alpha=alpha, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def tan(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Calculate an implementation-dependent approximation to the tangent, having
    domain ``(-infinity, +infinity)`` and codomain ``(-infinity, +infinity)``, for each
    element ``x_i`` of the input array ``x``. Each element ``x_i`` is assumed to be
    expressed in radians.
    .. note::
        Tangent is an analytical function on the complex plane
        and has no branch cuts. The function is periodic,
        with period :math:`\pi j`, with respect to the real
        component and has first order poles along the real
        line at coordinates :math:`(\pi (\frac{1}{2} + n), 0)`.
        However, IEEE 754 binary floating-point representation
        cannot represent the value :math:`\pi / 2` exactly, and,
        thus, no argument value is possible for
        which a pole error occurs.

        where :math:`{tanh}` is the hyperbolic tangent.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

    For complex floating-point operands, special cases must
    be handled as if the operation is implemented as ``-1j * tanh(x*1j)``.

    Parameters
    ----------
    x
        input array whose elements are expressed in radians. Should have a
        floating-point data type.
    out
        optional output, for writing the result to. It must have a shape that the inputs
        broadcast to.

    Returns
    -------
    ret
        an array containing the tangent of each element in ``x``. The return must have a
        floating-point data type determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.tan.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.tan(x)
    >>> print(y)
    ivy.array([0., 1.56, -2.19])

    >>> x = ivy.array([0.5, -0.7, 2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.tan(x, out=y)
    >>> print(y)
    ivy.array([0.546, -0.842, -0.916])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.tan(x, out=x)
    >>> print(x)
    ivy.array([[1.96, -1.37, 0.16],
        [-3.1, 0.996, -0.328]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.tan(x)
    >>> print(y)
    {
        a: ivy.array([0., 1.56, -2.19]),
        b: ivy.array([-0.143, 1.16, -3.38])
    }
    """
    return ivy.current_backend(x).tan(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def tanh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate an implementation-dependent approximation to the hyperbolic tangent,
    having domain ``[-infinity, +infinity]`` and codomain ``[-1, +1]``, for each
    element ``x_i`` of the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+1``.
    - If ``x_i`` is ``-infinity``, the result is ``-1``.

    For complex floating-point operands, let ``a = real(x_i)``,
    ``b = imag(x_i)``, and

    .. note::
       For complex floating-point operands, ``tanh(conj(x))``
       must equal ``conj(tanh(x))``.

    - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
    - If ``a`` is a nonzero finite number and ``b`` is
      ``+infinity``, the result is ``NaN + NaN j``.
    - If ``a`` is ``+0`` and ``b`` is ``+infinity``,
      the result is ``+0 + NaN j``.
    - If ``a`` is a nonzero finite number and ``b``
      is ``NaN``, the result is ``NaN + NaN j``.
    - If ``a`` is ``+0`` and ``b`` is ``NaN``,
      the result is ``+0 + NaN j``.
    - If ``a`` is ``+infinity`` and ``b`` is a positive
      (i.e., greater than ``0``) finite number, the result is ``1 + 0j``.
    - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``,
      the result is ``1 + 0j`` (sign of the imaginary
      component is unspecified).
    - If ``a`` is ``+infinity`` and ``b`` is ``NaN``,
      the result is ``1 + 0j`` (sign of the imaginary
      component is unspecified).
    - If ``a`` is ``NaN`` and ``b`` is ``+0``,
      the result is ``NaN + 0j``.
    - If ``a`` is ``NaN`` and ``b`` is a nonzero number,
      the result is ``NaN + NaN j``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``,
      the result is ``NaN + NaN j``.

    .. warning::
       For historical reasons stemming from the C standard,
       array libraries may not return the expected
       result when ``a`` is ``+0`` and ``b`` is either
       ``+infinity`` or ``NaN``. The result should be
       ``+0 + NaN j`` in both cases; however, for libraries
       compiled against older C versions, the result may be
       ``NaN + NaN j``.

       Array libraries are not required to patch these older
       C versions, and, thus, users are advised that results
       may vary across array library implementations for
       these special cases.


    Parameters
    ----------
    x
        input array whose elements each represent a hyperbolic angle. Should have a
        real-valued floating-point data
        type.
    out
        optional output, for writing the result to. It must have a shape that the inputs
        broadcast to.

    Returns
    -------
    ret
        an array containing the hyperbolic tangent of each element in ``x``.
        The returned array must have a real-valued floating-point data type
        determined by :ref:`type-promotion`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.tanh.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.tanh(x)
    >>> print(y)
    ivy.array([0., 0.762, 0.964])

    >>> x = ivy.array([0.5, -0.7, 2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.tanh(x, out=y)
    >>> print(y)
    ivy.array([0.462, -0.604, 0.984])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.tanh(x, out=x)
    >>> print(x)
    ivy.array([[0.8, 0.976, 0.997],
              [-1., -1., -1.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> y = ivy.tanh(x)
    >>> print(y)
    {
        a: ivy.array([0., 0.762, 0.964]),
        b: ivy.array([0.995, 0.999, 1.])
    }
    """
    return ivy.current_backend(x).tanh(x, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def trapz(
    y: ivy.Array,
    /,
    *,
    x: Optional[ivy.Array] = None,
    dx: float = 1.0,
    axis: int = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Integrate along the given axis using the composite trapezoidal rule.

    If x is provided, the integration happens in sequence along its elements
    - they are not sorted..

    Parameters
    ----------
    y
        The array that should be integrated.
    x
        The sample points corresponding to the input array values.
        If x is None, the sample points are assumed to be evenly spaced
        dx apart. The default is None.
    dx
        The spacing between sample points when x is None. The default is 1.
    axis
        The axis along which to integrate.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Definite integral of n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If the input array is a
        1-dimensional array, then the result is a float. If n is greater
        than 1, then the result is an n-1 dimensional array.

    Examples
    --------
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.trapz([1,2,3])
    4.0
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.trapz([1,2,3], x=[4, 6, 8])
    8.0
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.trapz([1,2,3], dx=2)
    8.0
    """
    return ivy.current_backend(y).trapz(y, x=x, dx=dx, axis=axis, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def trunc(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Round each element x_i of the input array x to the integer-valued number that is
    closest to but no greater than x_i.

    **Special cases**

    - If ``x_i`` is already an integer-valued, the result is ``x_i``.

    For floating-point operands,

    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``NaN``, the result is ``NaN``.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rounded result for each element in ``x``.
        The returned array must have the same data type as ``x``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.trunc.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1, 0.54, 3.67, -0.025])
    >>> y = ivy.trunc(x)
    >>> print(y)
    ivy.array([-1.,  0.,  3., -0.])

    >>> x = ivy.array([0.56, 7, -23.4, -0.0375])
    >>> ivy.trunc(x, out=x)
    >>> print(x)
    ivy.array([  0.,   7., -23.,  -0.])

    >>> x = ivy.array([[0.4, -8, 0.55], [0, 0.032, 2]])
    >>> y = ivy.zeros([2,3])
    >>> ivy.trunc(x, out=y)
    >>> print(y)
    ivy.array([[ 0., -8.,  0.],
           [ 0.,  0.,  2.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-0.25, 4, 1.3]), b=ivy.array([12, -3.5, 1.234]))
    >>> y = ivy.trunc(x)
    >>> print(y)
    {
        a: ivy.array([-0., 4., 1.]),
        b: ivy.array([12., -3., 1.])
    }
    """
    return ivy.current_backend(x).trunc(x, out=out)


# Extra #
# ------#


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def erf(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Gauss error function of ``x`` element-wise.

    Parameters
    ----------
    x
        Value to compute exponential for.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The Gauss error function of x.

    Examples
    --------
    >>> x = ivy.array([0, 0.3, 0.7, 1.0])
    >>> ivy.erf(x)
    ivy.array([0., 0.328, 0.677, 0.842])
    """
    return ivy.current_backend(x).erf(x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def maximum(
    x1: Union[ivy.Array, ivy.NativeArray, Number],
    x2: Union[ivy.Array, ivy.NativeArray, Number],
    /,
    *,
    use_where: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the max of x1 and x2 (i.e. x1 > x2 ? x1 : x2) element-wise.

    Parameters
    ----------
    x1
        Input array containing elements to maximum threshold.
    x2
        Tensor containing maximum values, must be broadcastable to x1.
    use_where
        Whether to use :func:`where` to calculate the maximum. If ``False``, the maximum
        is calculated using the ``(x + y + |x - y|)/2`` formula. Default is ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the elements of x1, but clipped to not be lower than the x2
        values.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([7, 9, 5])
    >>> y = ivy.array([9, 3, 2])
    >>> z = ivy.maximum(x, y)
    >>> print(z)
    ivy.array([9, 9, 5])

    >>> x = ivy.array([1, 5, 9, 8, 3, 7])
    >>> y = ivy.array([[9], [3], [2]])
    >>> z = ivy.zeros((3, 6))
    >>> ivy.maximum(x, y, out=z)
    >>> print(z)
    ivy.array([[9.,9.,9.,9.,9.,9.],
               [3.,5.,9.,8.,3.,7.],
               [2.,5.,9.,8.,3.,7.]])

    >>> x = ivy.array([[7, 3]])
    >>> y = ivy.array([0, 7])
    >>> ivy.maximum(x, y, out=x)
    >>> print(x)
    ivy.array([[7, 7]])

    With one :class:`ivy.Container` input:

    >>> x = ivy.array([[1, 3], [2, 4], [3, 7]])
    >>> y = ivy.Container(a=ivy.array([1, 0,]),
    ...                   b=ivy.array([-5, 9]))
    >>> z = ivy.maximum(x, y)
    >>> print(z)
    {
        a: ivy.array([[1, 3],
                      [2, 4],
                      [3, 7]]),
        b: ivy.array([[1, 9],
                      [2, 9],
                      [3, 9]])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([1, 3, 1]),b=ivy.array([2, 8, 5]))
    >>> y = ivy.Container(a=ivy.array([1, 5, 6]),b=ivy.array([5, 9, 7]))
    >>> z = ivy.maximum(x, y)
    >>> print(z)
    {
        a: ivy.array([1, 5, 6]),
        b: ivy.array([5, 9, 7])
    }
    """
    return ivy.current_backend(x1).maximum(x1, x2, use_where=use_where, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def minimum(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    use_where: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the min of x1 and x2 (i.e. x1 < x2 ? x1 : x2) element-wise.

    Parameters
    ----------
    x1
        Input array containing elements to minimum threshold.
    x2
        Tensor containing minimum values, must be broadcastable to x1.
    use_where
        Whether to use :func:`where` to calculate the minimum. If ``False``, the minimum
        is calculated using the ``(x + y - |x - y|)/2`` formula. Default is ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the elements of x1, but clipped to not exceed the x2 values.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([7, 9, 5])
    >>> y = ivy.array([9, 3, 2])
    >>> z = ivy.minimum(x, y)
    >>> print(z)
    ivy.array([7, 3, 2])

    >>> x = ivy.array([1, 5, 9, 8, 3, 7])
    >>> y = ivy.array([[9], [3], [2]])
    >>> z = ivy.zeros((3, 6))
    >>> ivy.minimum(x, y, out=z)
    >>> print(z)
    ivy.array([[1.,5.,9.,8.,3.,7.],
               [1.,3.,3.,3.,3.,3.],
               [1.,2.,2.,2.,2.,2.]])

    >>> x = ivy.array([[7, 3]])
    >>> y = ivy.array([0, 7])
    >>> ivy.minimum(x, y, out=x)
    >>> print(x)
    ivy.array([[0, 3]])

    With one :class:`ivy.Container` input:

    >>> x = ivy.array([[1, 3], [2, 4], [3, 7]])
    >>> y = ivy.Container(a=ivy.array([1, 0,]),b=ivy.array([-5, 9]))
    >>> z = ivy.minimum(x, y)
    >>> print(z)
    {
        a: ivy.array([[1, 0],
                      [1, 0],
                      [1, 0]]),
        b: ivy.array([[-5, 3],
                      [-5, 4],
                      [-5, 7]])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([1, 3, 1]),
    ...                   b=ivy.array([2, 8, 5]))
    >>> y = ivy.Container(a=ivy.array([1, 5, 6]),
    ...                   b=ivy.array([5, 9, 7]))
    >>> z = ivy.minimum(x, y)
    >>> print(z)
    {
        a: ivy.array([1, 3, 1]),
        b: ivy.array([2, 8, 5])
    }
    """
    return ivy.current_backend(x1).minimum(x1, x2, use_where=use_where, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def reciprocal(
    x: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array with the reciprocal of each element in ``x``.

    Parameters
    ----------
    x
        Input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        A new array with the positive value of each element in ``x``.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.reciprocal(x)
    >>> print(y)
    ivy.array([1.        , 0.5       , 0.33333333])
    """
    return ivy.current_backend(x).reciprocal(x, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def deg2rad(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Convert the input from degrees to radians.

    Parameters
    ----------
    x
        input array whose elements are each expressed in degrees.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array with each element in ``x`` converted from degrees to radians.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x=ivy.array([0,90,180,270,360])
    >>> y=ivy.deg2rad(x)
    >>> print(y)
    ivy.array([0.  , 1.57, 3.14, 4.71, 6.28])

    >>> x=ivy.array([0,-1.5,-50,ivy.nan])
    >>> y=ivy.zeros(5)
    >>> ivy.deg2rad(x,out=y)
    >>> print(y)
    ivy.array([ 0.    , -0.0262, -0.873 ,     nan])

    >>> x = ivy.array([[1.1, 2.2, 3.3],[-4.4, -5.5, -6.6]])
    >>> ivy.deg2rad(x, out=x)
    >>> print(x)
    ivy.array([[ 0.0192,  0.0384,  0.0576],
        [-0.0768, -0.096 , -0.115 ]])


    >>> x=ivy.native_array([-0,20.1,ivy.nan])
    >>> y=ivy.zeros(3)
    >>> ivy.deg2rad(x,out=y)
    >>> print(y)
    ivy.array([0.   , 0.351,   nan])

    With :class:`ivy.Container` input:

    >>> x=ivy.Container(a=ivy.array([-0,20.1,-50.5,-ivy.nan]),
                        b=ivy.array([0,90,180,270,360]))
    >>> y=ivy.deg2rad(x)
    >>> print(y)
    {
        a: ivy.array([0., 0.351, -0.881, nan]),
        b: ivy.array([0., 1.57, 3.14, 4.71, 6.28])
    }

    >>> x=ivy.Container(a=ivy.array([0,90,180,270,360]),
    ...                 b=ivy.native_array([0,-1.5,-50,ivy.nan]))
    >>> y=ivy.deg2rad(x)
    >>> print(y)
    {
        a: ivy.array([0., 1.57, 3.14, 4.71, 6.28]),
        b: ivy.array([0., -0.0262, -0.873, nan])
    }
    """
    return ivy.current_backend(x).deg2rad(x, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def rad2deg(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Convert the input from radians to degrees.

    Parameters
    ----------
    x
        input array whose elements are each expressed in radians.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array with each element in ``x`` converted from radians to degrees.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x=ivy.array([0.,1.57,3.14,4.71,6.28])
    >>> y=ivy.rad2deg(x)
    >>> print(y)
    ivy.array([  0.,  90., 180., 270., 360.])

    >>> x=ivy.array([0.,-0.0262,-0.873,ivy.nan])
    >>> y=ivy.zeros(5)
    >>> ivy.rad2deg(x,out=y)
    >>> print(y)
    ivy.array([  0. ,  -1.5, -50. ,   nan])

    >>> x = ivy.array([[1.1, 2.2, 3.3],[-4.4, -5.5, -6.6]])
    >>> ivy.rad2deg(x, out=x)
    >>> print(x)
    ivy.array([[  63.,  126.,  189.],
        [-252., -315., -378.]])

    >>> x=ivy.native_array([-0,20.1,ivy.nan])
    >>> y=ivy.zeros(3)
    >>> ivy.rad2deg(x,out=y)
    >>> print(y)
    ivy.array([   0., 1150.,   nan])

    With :class:`ivy.Container` input:

    >>> x=ivy.Container(a=ivy.array([-0,20.1,-50.5,-ivy.nan]),
    ...                 b=ivy.array([0,1,2,3,4]))
    >>> y=ivy.rad2deg(x)
    >>> print(y)
    {
        a: ivy.array([0., 1150., -2890., nan]),
        b: ivy.array([0., 57.3, 115., 172., 229.])
    }

    >>> x=ivy.Container(a=ivy.array([0,10,180,8.5,6]),
    ...                 b=ivy.native_array([0,-1.5,0.5,ivy.nan]))
    >>> y=ivy.rad2deg(x)
    >>> print(y)
    {
        a: ivy.array([0., 573., 10300., 487., 344.]),
        b: ivy.array([0., -85.9, 28.6, nan])
    }
    """
    return ivy.current_backend(x).rad2deg(x, out=out)


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def trunc_divide(
    x1: Union[float, ivy.Array, ivy.NativeArray],
    x2: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Perform element-wise integer division of the inputs rounding the results towards
    zero.

    Parameters
    ----------
    x1
        dividend input array. Should have a numeric data type.
    x2
        divisor input array. Must be compatible with x1 (see Broadcasting). Should have
        a numeric data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the element-wise results. The returned array must have a
        floating-point data type determined by Type Promotion Rules.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x1 = ivy.array([2., 7., 9.])
    >>> x2 = ivy.array([3., -4., 0.6])
    >>> y = ivy.trunc_divide(x1, x2)
    >>> print(y)
    ivy.array([ 0., -1., 14.])
    """
    return ivy.trunc(ivy.divide(x1, x2), out=out)


trunc_divide.mixed_backend_wrappers = {
    "to_add": (
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def isreal(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Test each element ``x_i`` of the input array ``x`` to determine whether the
    element is real number.
    Returns a bool array, where True if input element is real.
    If element has complex type with zero complex part, the return value
    for that element is True.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing test results. An element ``out_i`` is ``True`` if ``x_i`` is
        real number and ``False`` otherwise. The returned array should have a data type
        of ``bool``.

    The descriptions above assume an array input for simplicity, but
    the method also accepts :class:`ivy.Container` instances in place of
    :class:`ivy.Array` or :class:`ivy.NativeArray` instances, as shown in the type hints
    and also the examples below.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([[[1.1], [float('inf')], [-6.3]]])
    >>> z = ivy.isreal(x)
    >>> print(z)
    ivy.array([[[True], [True], [True]]])

    >>> x = ivy.array([1-0j, 3j, 7+5j])
    >>> z = ivy.isreal(x)
    >>> print(z)
    ivy.array([ True, False, False])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-6.7-7j, -np.inf, 1.23]),\
                          b=ivy.array([5j, 5-6j, 3]))
    >>> z = ivy.isreal(x)
    >>> print(z)
    {
        a: ivy.array([False, True, True]),
        b: ivy.array([False, False, True])
    }
    """
    return ivy.current_backend(x).isreal(x, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def fmod(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Compute the element-wise remainder of divisions of two arrays.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with element-wise remainder of divisions.

    Examples
    --------
    >>> x1 = ivy.array([2, 3, 4])
    >>> x2 = ivy.array([1, 5, 2])
    >>> ivy.fmod(x1, x2)
    ivy.array([ 0,  3,  0])

    >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
    >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
    >>> ivy.fmod(x1, x2)
    ivy.array([ nan,  nan,  nan])
    """
    return ivy.current_backend(x1, x2).fmod(x1, x2, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def lcm(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the element-wise least common multiple (LCM) of x1 and x2.

    Parameters
    ----------
    x1
        first input array, must be integers
    x2
        second input array, must be integers
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        an array that includes the element-wise least common multiples of x1 and x2

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x1=ivy.array([2, 3, 4])
    >>> x2=ivy.array([5, 7, 15])
    >>> x1.lcm(x1, x2)
    ivy.array([10, 21, 60])
    """
    return ivy.current_backend(x1, x2).lcm(x1, x2, out=out)
