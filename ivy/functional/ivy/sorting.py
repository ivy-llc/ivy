# global
from typing import Union, Optional

# local
import ivy
from ivy.backend_handler import current_backend as _cur_backend


# Array API Standard #
# -------------------#


def argsort(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Returns the indices that sort an array ``x`` along a specified axis.

    Parameters
    ----------
    x
        input array.
    axis
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending
        sort order. If ``True``, the returned indices sort ``x`` in descending order
        (by value). If ``False``, the returned indices sort ``x`` in ascending order
        (by value). Default: ``False``.
    stable
        sort stability. If ``True``, the returned indices must maintain the relative
        order of ``x`` values which compare as equal. If ``False``, the returned indices
        may or may not maintain the relative order of ``x`` values which compare as
        equal (i.e., the relative order of ``x`` values which compare as equal is
        implementation-dependent). Default: ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array of indices. The returned array must have the same shape as ``x``. The
        returned array must have the default array index data type.

    """
    return _cur_backend(x).argsort(x, axis, descending, stable, out)


def sort(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Returns a sorted copy of an array.

    Parameters
    ----------
    x
        input array
    axis
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending
        direction  The direction in which to sort the values
    stable
        sort stability. If ``True``,
        the returned indices must maintain the relative order of ``x`` values which
        compare as equal. If ``False``, the returned indices may or may not maintain the
        relative order of ``x`` values which compare as equal (i.e., the relative order
        of ``x`` values which compare as equal is implementation-dependent).
        Default: ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the same dtype and shape as `values`, with the elements sorted
        along the given `axis`.

    """
    return _cur_backend(x).sort(x, axis, descending, stable, out)


# Extra #
# ------#
