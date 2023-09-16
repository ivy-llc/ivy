# global
import tensorflow as tf
from typing import Union, Optional, Literal, List

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def argsort(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    x: Union[tf.Tensor, tf.Variable], (required)
        The input tf.Tensor or tf.Variable array to be sorted
    axis: int, (optional, default=-1)
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending: bool, (optional, default=False)
        direction  The direction in which to sort the values
    stable: bool, (optional, default=True)
        sort stability. If ``True``,
        the returned indices must maintain the relative order of ``x`` values which
        compare as equal. If ``False``, the returned indices may or may not maintain the
        relative order of ``x`` values which compare as equal (i.e., the relative order
        of ``x`` values which compare as equal is implementation-dependent).
        Default: ``True``.
    out: Optional[np.ndarray], (optional, default=None)
        optional output array, for writing the result to. It must have the same shape
        as ``x``.


    Returns
    -------
    ret
        Returns a new TensorFlow tensor or variable containing the indices that would
        sort the elements of the input tensor x based on the specified sorting criteria.


    Examples
    --------
    With :class:`np.ndarray` input:

    >>> x = tf.constant([[3, 1], [2, 4]], dtype=tf.int32)
    >>> y = tensorflow.argsort(x)
    >>> print(y)
    <tf.Tensor: shape=(2, 2), dtype=int64, numpy=array([[1, 0], [0, 1]])>

    >>> x = tf.constant([3, 1, 4, 1, 5], dtype=tf.int32)
    >>> y = tensorflow.argsort(x, descending=True)
    >>> print(y)
    <tf.Tensor: shape=(5,), dtype=int64, numpy=array([4, 2, 0, 3, 1])>

    >>> x = tf.constant([3, 1, 4, 1, 5], dtype=tf.int32)
    >>> tensorflow.argsort(x, axis=0, descending=True, stable=False, out=x)
    >>> print(x)
    <tf.Tensor: shape=(5,), dtype=int64, numpy=array([1, 3, 0, 2, 4])>
    """
    direction = "DESCENDING" if descending else "ASCENDING"
    x = tf.convert_to_tensor(x)
    is_bool = x.dtype.is_bool
    if is_bool:
        x = tf.cast(x, tf.int32)
    ret = tf.argsort(x, axis=axis, direction=direction, stable=stable)
    return tf.cast(ret, dtype=tf.int64)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def sort(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    x: Union[tf.Tensor, tf.Variable], (required)
        The input tf.Tensor or tf.Variable array to be sorted
    axis: int, (optional, default=-1)
       axis along which to sort. If set to ``-1``, the function must sort along the
       last axis. Default: ``-1``.
    descending: bool, (optional, default=False)
       direction  The direction in which to sort the values
    stable: bool, (optional, default=True)
       sort stability. If ``True``,
       the returned indices must maintain the relative order of ``x`` values which
       compare as equal. If ``False``, the returned indices may or may not maintain the
       relative order of ``x`` values which compare as equal (i.e., the relative order
       of ``x`` values which compare as equal is implementation-dependent).
       Default: ``True``.
    out: Optional[np.ndarray], (optional, default=None)
       optional output array, for writing the result to. It must have the same shape
       as ``x``.

    Returns
    -------
    ret
        Returns a new TensorFlow tensor or variable containing the sorted elements
        from the input tensor x based on the specified sorting criteria.


    Examples
    --------
    With :class:`np.ndarray` input:

    >>> x = tf.constant([[3, 1], [2, 4]], dtype=tf.int32)
    >>> y = tensorflow.sort(x)
    >>> print(y)
    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=array([[1, 3],[2, 4]], dtype=int32)>

    >>> x = tf.constant([3, 1, 4, 1, 5], dtype=tf.int32)
    >>> y = tensorflow.sort(x, axis=1, descending=True, stable=False)
    >>> print(y)
    <tf.Tensor: shape=(5,), dtype=int64, numpy=array([4, 2, 0, 3, 1])>

    >>> x = tf.constant([3, 1, 4, 1, 5], dtype=tf.int32)
    >>> y = tensorflow.zeros(5)
    >>> tensorflow.sort(x, descending=True, stable=False, out=y)
    >>> print(y)
    <tf.Tensor: shape=(5,), dtype=int32, numpy=array([1, 1, 3, 4, 5], dtype=int32)>
    """
    # TODO: handle stable sort when it's supported in tensorflow
    # currently it supports only quicksort (unstable)
    direction = "DESCENDING" if descending else "ASCENDING"
    x = tf.convert_to_tensor(x)
    is_bool = x.dtype.is_bool
    if is_bool:
        x = tf.cast(x, tf.int32)
    ret = tf.sort(x, axis=axis, direction=direction)
    if is_bool:
        ret = tf.cast(ret, dtype=tf.bool)
    return ret


# msort
@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def msort(
    a: Union[tf.Tensor, tf.Variable, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    """
    Return a copy of an array sorted along the first axis.

    Parameters
    ----------
    a: Union[tf.Tensor, tf.Variable, list, tuple], (required)
        The input array-like object (tf.Tensor, tf.Variable, list, or tuple)
        that you want to sort. array-like input.
    out:  Optional[Union[tf.Tensor, tf.Variable]], (optional, default=None)
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns a new TensorFlow tensor or variable containing the sorted
        elements from the input data structure a. The sorting is performed
        in ascending order along the first axis.

    Examples
    --------
    >>> a = tf.constant([[3, 1], [2, 4]], dtype=tf.int32)
    >>> tensorflow.msort(a)
    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=array([[2, 1], [3, 4]], dtype=int32)>
    """
    return tf.sort(a, axis=0)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def searchsorted(
    x: Union[tf.Tensor, tf.Variable],
    v: Union[tf.Tensor, tf.Variable],
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Optional[Union[ivy.Array, ivy.NativeArray, List[int]]] = None,
    ret_dtype: tf.DType = tf.int64,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    """
    Return the indices of the inserted elements in a sorted array.

    Parameters
    ----------
    x: Union[tf.Tensor, tf.Variable], (required)
        The input tf.Tensor or tf.Variable array to be sorted
        Input array. If `sorter` is None, then it must be sorted in ascending order,
        otherwise `sorter` must be an array of indices that sort it.
    v: Union[tf.Tensor, tf.Variable], (required)
        specific elements to insert in array x1
    side: Literal["left", "right"], (optional, default=left)
        The specific elements' index is at the 'left' side or
        'right' side in the sorted array x1. If the side is 'left', the
        index of the first suitable location located is given. If
        'right', return the last such index.
    sorter: Optional[Union[np.ndarray, List[int]]], (optional, default=None)
        optional array of integer indices that sort array x into ascending order,
        typically the result of argsort.
    ret_dtype: np.dtype, (optional, default=np.int64)
        the data type for the return value, Default: ivy.int64,
        only integer data types is allowed.
    out: Optional[np.ndarray], (optional, default=None)
        optional output array, for writing the result to.


    Returns
    -------
    Returns a new TensorFlow tensor or variable containing the indices where
    the values from v should be inserted into the sorted tensor or variable x
    to maintain the sorted order.

    Examples
    --------
    With :class:`np.ndarray` input:

    >>> x = tf.constant([1, 2, 4, 6, 8], dtype=tf.int32)
    >>> v = tf.constant([3, 5, 7], dtype=tf.int32)
    >>> y  = tensorflow.searchsorted(x, v)
    >>> print(y)
    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 3, 4])>


    >>> x = tf.constant([1, 2, 2, 4, 6, 8], dtype=tf.int32)
    >>> v = tf.constant([2], dtype=tf.int32)
    >>> y  = tensorflow.searchsorted(x, v, side='right')
    >>> print(y)
    <tf.Tensor: shape=(1,), dtype=int64, numpy=array([3])>

    >>> x = tf.constant([5, 2, 8, 1, 6, 3], dtype=tf.int32)
    >>> v = tf.constant([3, 5, 7], dtype=tf.int32)
    >>> y  = tensorflow.searchsorted(x, v)
    >>> print(y)
    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 0, 5])>
    """

    assert ivy.is_int_dtype(ret_dtype), ValueError(
        "only Integer data types are supported for ret_dtype."
    )
    is_supported_int_ret_dtype = ret_dtype in [tf.int32, tf.int64]
    if sorter is not None:
        assert ivy.is_int_dtype(sorter.dtype) and not ivy.is_uint_dtype(
            sorter.dtype
        ), TypeError(
            f"Only signed integer data type for sorter is allowed, got {sorter.dtype}."
        )
        if sorter.dtype not in [tf.int32, tf.int64]:
            sorter = tf.cast(sorter, tf.int32)
        if len(x.shape) == 1:
            x = tf.gather(x, sorter)
        else:
            x = tf.gather(x, sorter, batch_dims=-1)
    if len(x.shape) == 1 and len(v.shape) != 1:
        out_shape = v.shape
        v = tf.reshape(v, (1, -1))  # Leading dims must be the same
        if is_supported_int_ret_dtype:
            return tf.reshape(
                tf.searchsorted(x, v, side=side, out_type=ret_dtype), out_shape
            )
        else:
            return tf.cast(
                tf.reshape(tf.searchsorted(x, v, side=side), out_shape), ret_dtype
            )
    v = tf.cast(v, x.dtype)
    if is_supported_int_ret_dtype:
        return tf.searchsorted(x, v, side=side, out_type=ret_dtype)
    return tf.cast(tf.searchsorted(x, v, side=side), ret_dtype)
