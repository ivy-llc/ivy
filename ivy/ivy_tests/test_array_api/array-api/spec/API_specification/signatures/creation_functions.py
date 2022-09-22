from ._types import (List, NestedSequence, Optional, SupportsBufferProtocol, Tuple, Union, array,
                     device, dtype)

def arange(start: Union[int, float], /, stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional array.

    Parameters
    ----------
    start: Union[int, float]
        if ``stop`` is specified, the start of interval (inclusive); otherwise, the end of the interval (exclusive). If ``stop`` is not specified, the default starting value is ``0``.
    stop: Optional[Union[int, float]]
        the end of the interval. Default: ``None``.
    step: Union[int, float]
        the distance between two adjacent elements (``out[i+1] - out[i]``). Must not be ``0``; may be negative, this results in an empty array if ``stop >= start``. Default: ``1``.
    dtype: Optional[dtype]
        output array data type. Should be a floating-point data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. Default: ``None``.


    .. note::
       This function cannot guarantee that the interval does not include the ``stop`` value in those cases where ``step`` is not an integer and floating-point rounding errors affect the length of the output array.

    .. note::
       While this specification recommends that this function only return arrays having a floating-point data type, specification-compliant array libraries may choose to support output arrays having an integer data type (e.g., due to backward compatibility concerns). However, function behavior when generating integer output arrays is unspecified and, thus, is implementation-defined. Accordingly, using this function to generate integer output arrays is not portable.

    .. note::
       As mixed data type promotion is implementation-defined, behavior when ``start`` or ``stop`` exceeds the maximum safe integer of an output floating-point data type is implementation-defined. An implementation may choose to overflow or raise an exception.

    Returns
    -------
    out: array
        a one-dimensional array containing evenly spaced values. The length of the output array must be ``ceil((stop-start)/step)`` if ``stop - start`` and ``step`` have the same sign, and length ``0`` otherwise.
    """

def asarray(obj: Union[array, bool, int, float, NestedSequence, SupportsBufferProtocol], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None, copy: Optional[bool] = None) -> array:
    """
    Convert the input to an array.

    Parameters
    ----------
    obj: Union[array, bool, int, float, NestedSequence[bool | int | float], SupportsBufferProtocol]
        object to be converted to an array. May be a Python scalar, a (possibly nested) sequence of Python scalars, or an object supporting the Python buffer protocol.

        .. admonition:: Tip
           :class: important

           An object supporting the buffer protocol can be turned into a memoryview through ``memoryview(obj)``.

    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from the data type(s) in ``obj``. If all input values are Python scalars, then

        -   if all values are of type ``bool``, the output data type must be ``bool``.
        -   if the values are a mixture of ``bool``\s and ``int``, the output data type must be the default integer data type.
        -   if one or more values are ``float``\s, the output data type must be the default floating-point data type.

        Default: ``None``.

        .. admonition:: Note
           :class: note

           If ``dtype`` is not ``None``, then array conversions should obey :ref:`type-promotion` rules. Conversions not specified according to :ref:`type-promotion` rules may or may not be permitted by a conforming array library. To perform an explicit cast, use :func:`signatures.data_type_functions.astype`.

        .. note::
           If an input value exceeds the precision of the resolved output array data type, behavior is left unspecified and, thus, implementation-defined.

    device: Optional[device]
        device on which to place the created array. If ``device`` is ``None`` and ``x`` is an array, the output array device must be inferred from ``x``. Default: ``None``.
    copy: Optional[bool]
        boolean indicating whether or not to copy the input. If ``True``, the function must always copy. If ``False``, the function must never copy for input which supports the buffer protocol and must raise a ``ValueError`` in case a copy would be necessary. If ``None``, the function must reuse existing memory buffer if possible and copy otherwise. Default: ``None``.

    Returns
    -------
    out: array
        an array containing the data from ``obj``.
    """

def empty(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns an uninitialized array having a specified `shape`.

    Parameters
    ----------
    shape: Union[int, Tuple[int, ...]]
        output array shape.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out: array
        an array containing uninitialized data.
    """

def empty_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns an uninitialized array with the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the same shape as ``x`` and containing uninitialized data.
    """

def eye(n_rows: int, n_cols: Optional[int] = None, /, *, k: int = 0, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns a two-dimensional array with ones on the ``k``\th diagonal and zeros elsewhere.

    Parameters
    ----------
    n_rows: int
        number of rows in the output array.
    n_cols: Optional[int]
        number of columns in the output array. If ``None``, the default number of columns in the output array is equal to ``n_rows``. Default: ``None``.
    k: int
        index of the diagonal. A positive value refers to an upper diagonal, a negative value to a lower diagonal, and ``0`` to the main diagonal. Default: ``0``.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out: array
        an array where all elements are equal to zero, except for the ``k``\th diagonal, whose values are equal to one.
    """

def from_dlpack(x: object, /) -> array:
    """
    Returns a new array containing the data from another (array) object with a ``__dlpack__`` method.

    Parameters
    ----------
    x: object
        input (array) object.

    Returns
    -------
    out: array
        an array containing the data in `x`.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See :ref:`data-interchange` for details.
    """

def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[int, float], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns a new array having a specified ``shape`` and filled with ``fill_value``.

    Parameters
    ----------
    shape: Union[int, Tuple[int, ...]]
        output array shape.
    fill_value: Union[int, float]
        fill value.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``fill_value``. If the fill value is an ``int``, the output array data type must be the default integer data type. If the fill value is a ``float``, the output array data type must be the default floating-point data type. If the fill value is a ``bool``, the output array must have boolean data type. Default: ``None``.

        .. note::
           If the ``fill_value`` exceeds the precision of the resolved default output array data type, behavior is left unspecified and, thus, implementation-defined.

    device: Optional[device]
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out: array
        an array where every element is equal to ``fill_value``.
    """

def full_like(x: array, /, fill_value: Union[int, float], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns a new array filled with ``fill_value`` and having the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape.
    fill_value: Union[int, float]
        fill value.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.

        .. note::
           If the ``fill_value`` exceeds the precision of the resolved output array data type, behavior is unspecified and, thus, implementation-defined.

        .. note::
           If the ``fill_value`` has a data type (``int`` or ``float``) which is not of the same data type kind as the resolved output array data type (see :ref:`type-promotion`), behavior is unspecified and, thus, implementation-defined.

    device: Optional[device]
        device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the same shape as ``x`` and where every element is equal to ``fill_value``.
    """

def linspace(start: Union[int, float], stop: Union[int, float], /, num: int, *, dtype: Optional[dtype] = None, device: Optional[device] = None, endpoint: bool = True) -> array:
    """
    Returns evenly spaced numbers over a specified interval.

    Parameters
    ----------
    start: Union[int, float]
        the start of the interval.
    stop: Union[int, float]
        the end of the interval. If ``endpoint`` is ``False``, the function must generate a sequence of ``num+1`` evenly spaced numbers starting with ``start`` and ending with ``stop`` and exclude the ``stop`` from the returned array such that the returned array consists of evenly spaced numbers over the half-open interval ``[start, stop)``. If ``endpoint`` is ``True``, the output array must consist of evenly spaced numbers over the closed interval ``[start, stop]``. Default: ``True``.

        .. note::
           The step size changes when `endpoint` is `False`.

    num: int
        number of samples. Must be a non-negative integer value; otherwise, the function must raise an exception.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. Default: ``None``.
    endpoint: bool
        boolean indicating whether to include ``stop`` in the interval. Default: ``True``.

    Returns
    -------
    out: array
        a one-dimensional array containing evenly spaced values.
    """

def meshgrid(*arrays: array, indexing: str = 'xy') -> List[array]:
    """
    Returns coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays: array
        an arbitrary number of one-dimensional arrays representing grid coordinates. Each array should have the same numeric data type.
    indexing: str
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases, respectively), the ``indexing`` keyword has no effect and should be ignored. Default: ``'xy'``.

    Returns
    -------
    out: List[array]
        list of N arrays, where ``N`` is the number of provided one-dimensional input arrays. Each returned array must have rank ``N``. For ``N`` one-dimensional arrays having lengths ``Ni = len(xi)``,

        - if matrix indexing ``ij``, then each returned array must have the shape ``(N1, N2, N3, ..., Nn)``.
        - if Cartesian indexing ``xy``, then each returned array must have shape ``(N2, N1, N3, ..., Nn)``.

        Accordingly, for the two-dimensional case with input one-dimensional arrays of length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array must have shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned array must have shape ``(N, M)``.

        Similarly, for the three-dimensional case with input one-dimensional arrays of length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned array must have shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then each returned array must have shape ``(N, M, P)``.

        Each returned array should have the same data type as the input arrays.
    """

def ones(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns a new array having a specified ``shape`` and filled with ones.

    Parameters
    ----------
    shape: Union[int, Tuple[int, ...]]
        output array shape.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out: array
        an array containing ones.
    """

def ones_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns a new array filled with ones and having the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the same shape as ``x`` and filled with ones.
    """

def tril(x: array, /, *, k: int = 0) -> array:
    """
    Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::
       The lower triangular part of the matrix is defined as the elements on and below the specified diagonal ``k``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
    k: int
        diagonal above which to zero elements. If ``k = 0``, the diagonal is the main diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.

        .. note::
           The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on the interval ``[0, min(M, N) - 1]``.

    Returns
    -------
    out: array
        an array containing the lower triangular part(s). The returned array must have the same shape and data type as ``x``. All elements above the specified diagonal ``k`` must be zeroed. The returned array should be allocated on the same device as ``x``.
    """

def triu(x: array, /, *, k: int = 0) -> array:
    """
    Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::
       The upper triangular part of the matrix is defined as the elements on and above the specified diagonal ``k``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
    k: int
        diagonal below which to zero elements. If ``k = 0``, the diagonal is the main diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.

        .. note::
           The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on the interval ``[0, min(M, N) - 1]``.

    Returns
    -------
    out: array
        an array containing the upper triangular part(s). The returned array must have the same shape and data type as ``x``. All elements below the specified diagonal ``k`` must be zeroed. The returned array should be allocated on the same device as ``x``.
    """

def zeros(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape: Union[int, Tuple[int, ...]]
        output array shape.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out: array
        an array containing zeros.
    """

def zeros_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Returns a new array filled with zeros and having the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the same shape as ``x`` and filled with zeros.
    """

__all__ = ['arange', 'asarray', 'empty', 'empty_like', 'eye', 'from_dlpack', 'full', 'full_like', 'linspace', 'meshgrid', 'ones', 'ones_like', 'tril', 'triu', 'zeros', 'zeros_like']
