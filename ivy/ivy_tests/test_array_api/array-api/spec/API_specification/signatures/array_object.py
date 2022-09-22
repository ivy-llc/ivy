from __future__ import annotations

from ._types import (array, dtype as Dtype, device as Device, Optional, Tuple,
                     Union, Any, PyCapsule, Enum, ellipsis)

class _array():
    def __init__(self) -> None:
        """
        Initialize the attributes for the array object class.
        """

    @property
    def dtype() -> Dtype:
        """
        Data type of the array elements.

        Returns
        -------
        out: dtype
            array data type.
        """

    @property
    def device() -> Device:
        """
        Hardware device the array data resides on.

        Returns
        -------
        out: device
            a ``device`` object (see :ref:`device-support`).
        """

    @property
    def mT() -> array:
        """
        Transpose of a matrix (or a stack of matrices).

        If an array instance has fewer than two dimensions, an error should be raised.

        Returns
        -------
        out: array
            array whose last two dimensions (axes) are permuted in reverse order relative to original array (i.e., for an array instance having shape ``(..., M, N)``, the returned array must have shape ``(..., N, M)``). The returned array must have the same data type as the original array.
        """

    @property
    def ndim() -> int:
        """
        Number of array dimensions (axes).

        Returns
        -------
        out: int
            number of array dimensions (axes).
        """

    @property
    def shape() -> Tuple[Optional[int], ...]:
        """
        Array dimensions.

        Returns
        -------
        out: Tuple[Optional[int], ...]
            array dimensions. An array dimension must be ``None`` if and only if a dimension is unknown.


        .. note::
           For array libraries having graph-based computational models, array dimensions may be unknown due to data-dependent operations (e.g., boolean indexing; ``A[:, B > 0]``) and thus cannot be statically resolved without knowing array contents.

        .. note::
           The returned value should be a tuple; however, where warranted, an array library may choose to return a custom shape object. If an array library returns a custom shape object, the object must be immutable, must support indexing for dimension retrieval, and must behave similarly to a tuple.
        """

    @property
    def size() -> Optional[int]:
        """
        Number of elements in an array.

        .. note::
           This must equal the product of the array's dimensions.

        Returns
        -------
        out: Optional[int]
            number of elements in an array. The returned value must be ``None`` if and only if one or more array dimensions are unknown.


        .. note::
           For array libraries having graph-based computational models, an array may have unknown dimensions due to data-dependent operations.
        """

    @property
    def T() -> array:
        """
        Transpose of the array.

        The array instance must be two-dimensional. If the array instance is not two-dimensional, an error should be raised.

        Returns
        -------
        out: array
            two-dimensional array whose first and last dimensions (axes) are permuted in reverse order relative to original array. The returned array must have the same data type as the original array.


        .. note::
           Limiting the transpose to two-dimensional arrays (matrices) deviates from the NumPy et al practice of reversing all axes for arrays having more than two-dimensions. This is intentional, as reversing all axes was found to be problematic (e.g., conflicting with the mathematical definition of a transpose which is limited to matrices; not operating on batches of matrices; et cetera). In order to reverse all axes, one is recommended to use the functional ``permute_dims`` interface found in this specification.
        """

    def __abs__(self: array, /) -> array:
        """
        Calculates the absolute value for each element of an array instance (i.e., the element-wise result has the same magnitude as the respective element but has positive sign).

        .. note::
           For signed integer data types, the absolute value of the minimum representable integer is implementation-dependent.

        **Special cases**

        For floating-point operands, let ``self`` equal ``x``.

        -   If ``x_i`` is ``NaN``, the result is ``NaN``.
        -   If ``x_i`` is ``-0``, the result is ``+0``.
        -   If ``x_i`` is ``-infinity``, the result is ``+infinity``.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise absolute value. The returned array must have the same data type as ``self``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.abs`.
        """

    def __add__(self: array, other: Union[int, float, array], /) -> array:
        """
        Calculates the sum for each element of an array instance with the respective element of the array ``other``.

        **Special cases**

        For floating-point operands, let ``self`` equal ``x1`` and ``other`` equal ``x2``.

        -   If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``-infinity``, the result is ``NaN``.
        -   If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``+infinity``, the result is ``NaN``.
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``+infinity``, the result is ``+infinity``.
        -   If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``-infinity``, the result is ``-infinity``.
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is a finite number, the result is ``+infinity``.
        -   If ``x1_i`` is ``-infinity`` and ``x2_i`` is a finite number, the result is ``-infinity``.
        -   If ``x1_i`` is a finite number and ``x2_i`` is ``+infinity``, the result is ``+infinity``.
        -   If ``x1_i`` is a finite number and ``x2_i`` is ``-infinity``, the result is ``-infinity``.
        -   If ``x1_i`` is ``-0`` and ``x2_i`` is ``-0``, the result is ``-0``.
        -   If ``x1_i`` is ``-0`` and ``x2_i`` is ``+0``, the result is ``+0``.
        -   If ``x1_i`` is ``+0`` and ``x2_i`` is ``-0``, the result is ``+0``.
        -   If ``x1_i`` is ``+0`` and ``x2_i`` is ``+0``, the result is ``+0``.
        -   If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is a nonzero finite number, the result is ``x2_i``.
        -   If ``x1_i`` is a nonzero finite number and ``x2_i`` is either ``+0`` or ``-0``, the result is ``x1_i``.
        -   If ``x1_i`` is a nonzero finite number and ``x2_i`` is ``-x1_i``, the result is ``+0``.
        -   In the remaining cases, when neither ``infinity``, ``+0``, ``-0``, nor a ``NaN`` is involved, and the operands have the same mathematical sign or have different magnitudes, the sum must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported round mode. If the magnitude is too large to represent, the operation overflows and the result is an ``infinity`` of appropriate mathematical sign.

        .. note::
           Floating-point addition is a commutative operation, but not always associative.

        Parameters
        ----------
        self: array
            array instance (augend array). Should have a numeric data type.
        other: Union[int, float, array]
            addend array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise sums. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.add`.
        """

    def __and__(self: array, other: Union[int, bool, array], /) -> array:
        """
        Evaluates ``self_i & other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer or boolean data type.
        other: Union[int, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.bitwise_and`.
        """

    def __array_namespace__(self: array, /, *, api_version: Optional[str] = None) -> Any:
        """
        Returns an object that has all the array API functions on it.

        Parameters
        ----------
        self: array
            array instance.
        api_version: Optional[str]
            string representing the version of the array API specification to be returned, in ``'YYYY.MM'`` form, for example, ``'2020.10'``. If it is ``None``, it should return the namespace corresponding to latest version of the array API specification.  If the given version is invalid or not implemented for the given module, an error should be raised. Default: ``None``.

        Returns
        -------
        out: Any
            an object representing the array API namespace. It should have every top-level function defined in the specification as an attribute. It may contain other public names as well, but it is recommended to only include those names that are part of the specification.
        """

    def __bool__(self: array, /) -> bool:
        """
        Converts a zero-dimensional boolean array to a Python ``bool`` object.

        Parameters
        ----------
        self: array
            zero-dimensional array instance. Must have a boolean data type.

        Returns
        -------
        out: bool
            a Python ``bool`` object representing the single element of the array.
        """

    def __dlpack__(self: array, /, *, stream: Optional[Union[int, Any]] = None) -> PyCapsule:
        """
        Exports the array for consumption by :func:`~signatures.creation_functions.from_dlpack` as a DLPack capsule.

        Parameters
        ----------
        self: array
            array instance.
        stream: Optional[Union[int, Any]]
            for CUDA and ROCm, a Python integer representing a pointer to a stream, on devices that support streams. ``stream`` is provided by the consumer to the producer to instruct the producer to ensure that operations can safely be performed on the array (e.g., by inserting a dependency between streams via "wait for event"). The pointer must be a positive integer or ``-1``. If ``stream`` is ``-1``, the value may be used by the consumer to signal "producer must not perform any synchronization". The ownership of the stream stays with the consumer. On CPU and other device types without streams, only ``None`` is accepted.

            For other device types which do have a stream, queue or similar synchronization mechanism, the most appropriate type to use for ``stream`` is not yet determined. E.g., for SYCL one may want to use an object containing an in-order ``cl::sycl::queue``. This is allowed when libraries agree on such a convention, and may be standardized in a future version of this API standard.


        .. note::
            Support for a ``stream`` value other than ``None`` is optional and implementation-dependent.


        Device-specific notes:


        .. admonition:: CUDA
            :class: note

            - ``None``: producer must assume the legacy default stream (default).
            - ``1``: the legacy default stream.
            - ``2``: the per-thread default stream.
            - ``> 2``: stream number represented as a Python integer.
            - ``0`` is disallowed due to its ambiguity: ``0`` could mean either ``None``, ``1``, or ``2``.


        .. admonition:: ROCm
            :class: note

            - ``None``: producer must assume the legacy default stream (default).
            - ``0``: the default stream.
            - ``> 2``: stream number represented as a Python integer.
            - Using ``1`` and ``2`` is not supported.


        .. admonition:: Tip
            :class: important

            It is recommended that implementers explicitly handle streams. If
            they use the legacy default stream, specifying ``1`` (CUDA) or ``0``
            (ROCm) is preferred. ``None`` is a safe default for developers who do
            not want to think about stream handling at all, potentially at the
            cost of more synchronization than necessary.

        Returns
        -------
        capsule: PyCapsule
            a DLPack capsule for the array. See :ref:`data-interchange` for details.
        """

    def __dlpack_device__(self: array, /) -> Tuple[Enum, int]:
        """
        Returns device type and device ID in DLPack format. Meant for use within :func:`~signatures.creation_functions.from_dlpack`.

        Parameters
        ----------
        self: array
            array instance.

        Returns
        -------
        device: Tuple[Enum, int]
            a tuple ``(device_type, device_id)`` in DLPack format. Valid device type enum members are:

            ::

              CPU = 1
              CUDA = 2
              CPU_PINNED = 3
              OPENCL = 4
              VULKAN = 7
              METAL = 8
              VPI = 9
              ROCM = 10
        """

    def __eq__(self: array, other: Union[int, float, bool, array], /) -> array:
        """
        Computes the truth value of ``self_i == other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. May have any data type.
        other: Union[int, float, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). May have any data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.equal`.
        """

    def __float__(self: array, /) -> float:
        """
        Converts a zero-dimensional floating-point array to a Python ``float`` object.

        Parameters
        ----------
        self: array
            zero-dimensional array instance. Must have a floating-point data type.

        Returns
        -------
        out: float
            a Python ``float`` object representing the single element of the array instance.
        """

    def __floordiv__(self: array, other: Union[int, float, array], /) -> array:
        """
        Evaluates ``self_i // other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus implementation-defined.

        **Special cases**

        .. note::
            Floor division was introduced in Python via `PEP 238 <https://www.python.org/dev/peps/pep-0238/>`_ with the goal to disambiguate "true division" (i.e., computing an approximation to the mathematical operation of division) from "floor division" (i.e., rounding the result of division toward negative infinity). The former was computed when one of the operands was a ``float``, while the latter was computed when both operands were ``int``s. Overloading the ``/`` operator to support both behaviors led to subtle numerical bugs when integers are possible, but not expected.

            To resolve this ambiguity, ``/`` was designated for true division, and ``//`` was designated for floor division. Semantically, floor division was `defined <https://www.python.org/dev/peps/pep-0238/#semantics-of-floor-division>`_ as equivalent to ``a // b == floor(a/b)``; however, special floating-point cases were left ill-defined.

            Accordingly, floor division is not implemented consistently across array libraries for some of the special cases documented below. Namely, when one of the operands is ``infinity``, libraries may diverge with some choosing to strictly follow ``floor(a/b)`` and others choosing to pair ``//`` with ``%`` according to the relation ``b = a % b + b * (a // b)``. The special cases leading to divergent behavior are documented below.

            This specification prefers floor division to match ``floor(divide(x1, x2))`` in order to avoid surprising and unexpected results; however, array libraries may choose to more strictly follow Python behavior.

        For floating-point operands, let ``self`` equal ``x1`` and ``other`` equal ``x2``.

        -   If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        -   If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
        -   If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
        -   If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        -   If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``-0``.
        -   If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
        -   If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is ``+0``.
        -   If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is ``+infinity``.
        -   If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is ``-infinity``.
        -   If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is ``-infinity``.
        -   If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is ``+infinity``.
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity``. (**note**: libraries may return ``NaN`` to match Python behavior.)
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``-infinity``. (**note**: libraries may return ``NaN`` to match Python behavior.)
        -   If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``-infinity``. (**note**: libraries may return ``NaN`` to match Python behavior.)
        -   If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``+infinity``. (**note**: libraries may return ``NaN`` to match Python behavior.)
        -   If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``+0``.
        -   If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``-0``. (**note**: libraries may return ``-1.0`` to match Python behavior.)
        -   If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``-0``. (**note**: libraries may return ``-1.0`` to match Python behavior.)
        -   If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``+0``.
        -   If ``x1_i`` and ``x2_i`` have the same mathematical sign and are both nonzero finite numbers, the result has a positive mathematical sign.
        -   If ``x1_i`` and ``x2_i`` have different mathematical signs and are both nonzero finite numbers, the result has a negative mathematical sign.
        -   In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN`` is involved, the quotient must be computed and rounded to the greatest (i.e., closest to ``+infinity``) representable integer-value number that is not greater than the division result. If the magnitude is too large to represent, the operation overflows and the result is an ``infinity`` of appropriate mathematical sign. If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.floor_divide`.
        """

    def __ge__(self: array, other: Union[int, float, array], /) -> array:
        """
        Computes the truth value of ``self_i >= other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.greater_equal`.
        """

    def __getitem__(self: array, key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array], /) -> array:
        """
        Returns ``self[key]``.

        Parameters
        ----------
        self: array
            array instance.
        key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array]
            index key.

        Returns
        -------
        out: array
            an array containing the accessed value(s). The returned array must have the same data type as ``self``.
        """

    def __gt__(self: array, other: Union[int, float, array], /) -> array:
        """
        Computes the truth value of ``self_i > other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.greater`.
        """

    def __index__(self: array, /) -> int:
        """
        Converts a zero-dimensional integer array to a Python ``int`` object.

        .. note::
           This method is called to implement `operator.index() <https://docs.python.org/3/reference/datamodel.html#object.__index__>`_. See also `PEP 357 <https://www.python.org/dev/peps/pep-0357/>`_.

        Parameters
        ----------
        self: array
            zero-dimensional array instance. Must have an integer data type.

        Returns
        -------
        out: int
            a Python ``int`` object representing the single element of the array instance.
        """

    def __int__(self: array, /) -> int:
        """
        Converts a zero-dimensional integer array to a Python ``int`` object.

        Parameters
        ----------
        self: array
            zero-dimensional array instance. Must have an integer data type.

        Returns
        -------
        out: int
            a Python ``int`` object representing the single element of the array instance.
        """

    def __invert__(self: array, /) -> array:
        """
        Evaluates ``~self_i`` for each element of an array instance.

        Parameters
        ----------
        self: array
            array instance. Should have an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have the same data type as `self`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.bitwise_invert`.
        """

    def __le__(self: array, other: Union[int, float, array], /) -> array:
        """
        Computes the truth value of ``self_i <= other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.less_equal`.
        """

    def __lshift__(self: array, other: Union[int, array], /) -> array:
        """
        Evaluates ``self_i << other_i`` for each element of an array instance with the respective element  of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer data type.
        other: Union[int, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer data type. Each element must be greater than or equal to ``0``.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have the same data type as ``self``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.bitwise_left_shift`.
        """

    def __lt__(self: array, other: Union[int, float, array], /) -> array:
        """
        Computes the truth value of ``self_i < other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.less`.
        """

    def __matmul__(self: array, other: array, /) -> array:
        """
        Computes the matrix product.

        .. note::
           The ``matmul`` function must implement the same semantics as the built-in ``@`` operator (see `PEP 465 <https://www.python.org/dev/peps/pep-0465>`_).

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type. Must have at least one dimension. If ``self`` is one-dimensional having shape ``(M,)`` and ``other`` has more than one dimension, ``self`` must be promoted to a two-dimensional array by prepending ``1`` to its dimensions (i.e., must have shape ``(1, M)``). After matrix multiplication, the prepended dimensions in the returned array must be removed. If ``self`` has more than one dimension (including after vector-to-matrix promotion), ``shape(self)[:-2]`` must be compatible with ``shape(other)[:-2]`` (after vector-to-matrix promotion) (see :ref:`broadcasting`). If ``self`` has shape ``(..., M, K)``, the innermost two dimensions form matrices on which to perform matrix multiplication.
        other: array
            other array. Should have a numeric data type. Must have at least one dimension. If ``other`` is one-dimensional having shape ``(N,)`` and ``self`` has more than one dimension, ``other`` must be promoted to a two-dimensional array by appending ``1`` to its dimensions (i.e., must have shape ``(N, 1)``). After matrix multiplication, the appended dimensions in the returned array must be removed. If ``other`` has more than one dimension (including after vector-to-matrix promotion), ``shape(other)[:-2]`` must be compatible with ``shape(self)[:-2]`` (after vector-to-matrix promotion) (see :ref:`broadcasting`). If ``other`` has shape ``(..., K, N)``, the innermost two dimensions form matrices on which to perform matrix multiplication.

        Returns
        -------
        out: array
            -   if both ``self`` and ``other`` are one-dimensional arrays having shape ``(N,)``, a zero-dimensional array containing the inner product as its only element.
            -   if ``self`` is a two-dimensional array having shape ``(M, K)`` and ``other`` is a two-dimensional array having shape ``(K, N)``, a two-dimensional array containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ and having shape ``(M, N)``.
            -   if ``self`` is a one-dimensional array having shape ``(K,)`` and ``other`` is an array having shape ``(..., K, N)``, an array having shape ``(..., N)`` (i.e., prepended dimensions during vector-to-matrix promotion must be removed) and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_.
            -   if ``self`` is an array having shape ``(..., M, K)`` and ``other`` is a one-dimensional array having shape ``(K,)``, an array having shape ``(..., M)`` (i.e., appended dimensions during vector-to-matrix promotion must be removed) and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_.
            -   if ``self`` is a two-dimensional array having shape ``(M, K)`` and ``other`` is an array having shape ``(..., K, N)``, an array having shape ``(..., M, N)`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.
            -   if ``self`` is an array having shape ``(..., M, K)`` and ``other`` is a two-dimensional array having shape ``(K, N)``, an array having shape ``(..., M, N)`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.
            -   if either ``self`` or ``other`` has more than two dimensions, an array having a shape determined by :ref:`broadcasting` ``shape(self)[:-2]`` against ``shape(other)[:-2]`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.
            -   The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Results must equal the results returned by the equivalent function :func:`~signatures.linear_algebra_functions.matmul`.

        **Raises**

        - if either ``self`` or ``other`` is a zero-dimensional array.
        - if ``self`` is a one-dimensional array having shape ``(K,)``, ``other`` is a one-dimensional array having shape ``(L,)``, and ``K != L``.
        - if ``self`` is a one-dimensional array having shape ``(K,)``, ``other`` is an array having shape ``(..., L, N)``, and ``K != L``.
        - if ``self`` is an array having shape ``(..., M, K)``, ``other`` is a one-dimensional array having shape ``(L,)``, and ``K != L``.
        - if ``self`` is an array having shape ``(..., M, K)``, ``other`` is an array having shape ``(..., L, N)``, and ``K != L``.
        """

    def __mod__(self: array, other: Union[int, float, array], /) -> array:
        """
        Evaluates ``self_i % other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus implementation-defined.

        **Special Cases**

        .. note::
           In general, this method is **not** recommended for floating-point operands as semantics do not follow IEEE 754. That this method is specified to accept floating-point operands is primarily for reasons of backward compatibility.

        For floating-point operands, let ``self`` equal ``x1`` and ``other`` equal ``x2``.

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
        - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is ``NaN``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is ``NaN``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is ``NaN``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is ``NaN``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``NaN``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``NaN``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``NaN``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``NaN``.
        - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``x1_i``. (**note**: this result matches Python behavior.)
        - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``x2_i``. (**note**: this result matches Python behavior.)
        - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``x2_i``. (**note**: this results matches Python behavior.)
        - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``x1_i``. (**note**: this result matches Python behavior.)
        - In the remaining cases, the result must match that of the Python ``%`` operator.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. Each element-wise result must have the same sign as the respective element ``other_i``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.remainder`.
        """

    def __mul__(self: array, other: Union[int, float, array], /) -> array:
        """
        Calculates the product for each element of an array instance with the respective element of the array ``other``.

        **Special cases**

        For floating-point operands, let ``self`` equal ``x1`` and ``other`` equal ``x2``.

        -   If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        -   If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
        -   If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
        -   If ``x1_i`` and ``x2_i`` have the same mathematical sign, the result has a positive mathematical sign, unless the result is ``NaN``. If the result is ``NaN``, the "sign" of ``NaN`` is implementation-defined.
        -   If ``x1_i`` and ``x2_i`` have different mathematical signs, the result has a negative mathematical sign, unless the result is ``NaN``. If the result is ``NaN``, the "sign" of ``NaN`` is implementation-defined.
        -   If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        -   If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is a nonzero finite number, the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        -   If ``x1_i`` is a nonzero finite number and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        -   In the remaining cases, where neither ``infinity`` nor `NaN` is involved, the product must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode. If the magnitude is too large to represent, the result is an ``infinity`` of appropriate mathematical sign. If the magnitude is too small to represent, the result is a zero of appropriate mathematical sign.


        .. note::
           Floating-point multiplication is not always associative due to finite precision.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise products. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.multiply`.
        """

    def __ne__(self: array, other: Union[int, float, bool, array], /) -> array:
        """
        Computes the truth value of ``self_i != other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. May have any data type.
        other: Union[int, float, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). May have any data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool`` (i.e., must be a boolean array).


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.not_equal`.
        """

    def __neg__(self: array, /) -> array:
        """
        Evaluates ``-self_i`` for each element of an array instance.

        .. note::
           For signed integer data types, the numerical negative of the minimum representable integer is implementation-dependent.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the evaluated result for each element in ``self``. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.negative`.
        """

    def __or__(self: array, other: Union[int, bool, array], /) -> array:
        """
        Evaluates ``self_i | other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer or boolean data type.
        other: Union[int, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.bitwise_or`.
        """

    def __pos__(self: array, /) -> array:
        """
        Evaluates ``+self_i`` for each element of an array instance.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the evaluated result for each element. The returned array must have the same data type as ``self``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.positive`.
        """

    def __pow__(self: array, other: Union[int, float, array], /) -> array:
        """
        Calculates an implementation-dependent approximation of exponentiation by raising each element (the base) of an array instance to the power of ``other_i`` (the exponent), where ``other_i`` is the corresponding element of the array ``other``.

        .. note::
           If both ``self`` and ``other`` have integer data types, the result of ``__pow__`` when `other_i` is negative (i.e., less than zero) is unspecified and thus implementation-dependent.

           If ``self`` has an integer data type and ``other`` has a floating-point data type, behavior is implementation-dependent, as type promotion between data type "kinds" (e.g., integer versus floating-point) is unspecified.

        **Special cases**

        For floating-point operands, let ``self`` equal ``x1`` and ``other`` equal ``x2``.

        -   If ``x1_i`` is not equal to ``1`` and ``x2_i`` is ``NaN``, the result is ``NaN``.
        -   If ``x2_i`` is ``+0``, the result is ``1``, even if ``x1_i`` is ``NaN``.
        -   If ``x2_i`` is ``-0``, the result is `1`, even if ``x1_i`` is ``NaN``.
        -   If ``x1_i`` is ``NaN`` and ``x2_i`` is not equal to ``0``, the result is ``NaN``.
        -   If ``abs(x1_i)`` is greater than ``1`` and ``x2_i`` is ``+infinity``, the result is ``+infinity``.
        -   If ``abs(x1_i)`` is greater than ``1`` and ``x2_i`` is ``-infinity``, the result is ``+0``.
        -   If ``abs(x1_i)`` is ``1`` and ``x2_i`` is ``+infinity``, the result is ``1``.
        -   If ``abs(x1_i)`` is ``1`` and ``x2_i`` is ``-infinity``, the result is ``1``.
        -   If ``x1_i`` is ``1`` and ``x2_i`` is not ``NaN``, the result is ``1``.
        -   If ``abs(x1_i)`` is less than ``1`` and ``x2_i`` is ``+infinity``, the result is ``+0``.
        -   If ``abs(x1_i)`` is less than ``1`` and ``x2_i`` is ``-infinity``, the result is ``+infinity``.
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is greater than ``0``, the result is ``+infinity``.
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is less than ``0``, the result is ``+0``.
        -   If ``x1_i`` is ``-infinity``, ``x2_i`` is greater than ``0``, and ``x2_i`` is an odd integer value, the result is ``-infinity``.
        -   If ``x1_i`` is ``-infinity``, ``x2_i`` is greater than ``0``, and ``x2_i`` is not an odd integer value, the result is ``+infinity``.
        -   If ``x1_i`` is ``-infinity``, ``x2_i`` is less than ``0``, and ``x2_i`` is an odd integer value, the result is ``-0``.
        -   If ``x1_i`` is ``-infinity``, ``x2_i`` is less than ``0``, and ``x2_i`` is not an odd integer value, the result is ``+0``.
        -   If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        -   If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``+infinity``.
        -   If ``x1_i`` is ``-0``, ``x2_i`` is greater than ``0``, and ``x2_i`` is an odd integer value, the result is ``-0``.
        -   If ``x1_i`` is ``-0``, ``x2_i`` is greater than ``0``, and ``x2_i`` is not an odd integer value, the result is ``+0``.
        -   If ``x1_i`` is ``-0``, ``x2_i`` is less than ``0``, and ``x2_i`` is an odd integer value, the result is ``-infinity``.
        -   If ``x1_i`` is ``-0``, ``x2_i`` is less than ``0``, and ``x2_i`` is not an odd integer value, the result is ``+infinity``.
        -   If ``x1_i`` is less than ``0``, ``x1_i`` is a finite number, ``x2_i`` is a finite number, and ``x2_i`` is not an integer value, the result is ``NaN``.

        Parameters
        ----------
        self: array
            array instance whose elements correspond to the exponentiation base. Should have a numeric data type.
        other: Union[int, float, array]
            other array whose elements correspond to the exponentiation exponent. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.pow`.
        """

    def __rshift__(self: array, other: Union[int, array], /) -> array:
        """
        Evaluates ``self_i >> other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer data type.
        other: Union[int, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer data type. Each element must be greater than or equal to ``0``.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have the same data type as ``self``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.bitwise_right_shift`.
        """

    def __setitem__(self: array, key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array], value: Union[int, float, bool, array], /) -> None:
        """
        Sets ``self[key]`` to ``value``.

        Parameters
        ----------
        self: array
            array instance.
        key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array]
            index key.
        value: Union[int, float, bool, array]
            value(s) to set. Must be compatible with ``self[key]`` (see :ref:`broadcasting`).


        .. note::

           Setting array values must not affect the data type of ``self``.

           When ``value`` is a Python scalar (i.e., ``int``, ``float``, ``bool``), behavior must follow specification guidance on mixing arrays with Python scalars (see :ref:`type-promotion`).

           When ``value`` is an ``array`` of a different data type than ``self``, how values are cast to the data type of ``self`` is implementation defined.
        """

    def __sub__(self: array, other: Union[int, float, array], /) -> array:
        """
        Calculates the difference for each element of an array instance with the respective element of the array ``other``. The result of ``self_i - other_i`` must be the same as ``self_i + (-other_i)`` and must be governed by the same floating-point rules as addition (see :meth:`array.__add__`).

        Parameters
        ----------
        self: array
            array instance (minuend array). Should have a numeric data type.
        other: Union[int, float, array]
            subtrahend array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise differences. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.subtract`.
        """

    def __truediv__(self: array, other: Union[int, float, array], /) -> array:
        """
        Evaluates ``self_i / other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           If one or both of ``self`` and ``other`` have integer data types, the result is implementation-dependent, as type promotion between data type "kinds" (e.g., integer versus floating-point) is unspecified.

           Specification-compliant libraries may choose to raise an error or return an array containing the element-wise results. If an array is returned, the array must have a floating-point data type.

        **Special cases**

        For floating-point operands, let ``self`` equal ``x1`` and ``other`` equal ``x2``.

        -   If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        -   If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is `NaN`.
        -   If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
        -   If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        -   If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``-0``.
        -   If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
        -   If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is ``+0``.
        -   If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is ``+infinity``.
        -   If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is ``-infinity``.
        -   If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is ``-infinity``.
        -   If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is ``+infinity``.
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity``.
        -   If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``-infinity``.
        -   If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``-infinity``.
        -   If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``+infinity``.
        -   If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``+0``.
        -   If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``-0``.
        -   If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``-0``.
        -   If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``+0``.
        -   If ``x1_i`` and ``x2_i`` have the same mathematical sign and are both nonzero finite numbers, the result has a positive mathematical sign.
        -   If ``x1_i`` and ``x2_i`` have different mathematical signs and are both nonzero finite numbers, the result has a negative mathematical sign.
        -   In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN`` is involved, the quotient must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode. If the magnitude is too large to represent, the operation overflows and the result is an ``infinity`` of appropriate mathematical sign. If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array should have a floating-point data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.divide`.
        """

    def __xor__(self: array, other: Union[int, bool, array], /) -> array:
        """
        Evaluates ``self_i ^ other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer or boolean data type.
        other: Union[int, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~signatures.elementwise_functions.bitwise_xor`.
        """

    def to_device(self: array, device: Device, /, *, stream: Optional[Union[int, Any]] = None) -> array:
        """
        Copy the array from the device on which it currently resides to the specified ``device``.

        Parameters
        ----------
        self: array
            array instance.
        device: device
            a ``device`` object (see :ref:`device-support`).
        stream: Optional[Union[int, Any]]
            stream object to use during copy. In addition to the types supported in :meth:`array.__dlpack__`, implementations may choose to support any library-specific stream object with the caveat that any code using such an object would not be portable.

        Returns
        -------
        out: array
            an array with the same data and data type as ``self`` and located on the specified ``device``.


        .. note::
           If ``stream`` is given, the copy operation should be enqueued on the provided ``stream``; otherwise, the copy operation should be enqueued on the default stream/queue. Whether the copy is performed synchronously or asynchronously is implementation-dependent. Accordingly, if synchronization is required to guarantee data safety, this must be clearly explained in a conforming library's documentation.
        """

array = _array
