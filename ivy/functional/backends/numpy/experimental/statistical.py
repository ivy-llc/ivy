from typing import Optional, Union, Tuple, Sequence
import numpy as np

import ivy  # noqa
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version


def median(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if out is not None:
        out = np.reshape(out, input.shape)
    return np.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )


median.support_native_out = True


def nanmean(
    a: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


nanmean.support_native_out = True


@with_supported_dtypes({"1.23.0 and below": ("int32", "int64")}, backend_version)
def unravel_index(
    indices: np.ndarray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> Tuple:
    ret = np.asarray(np.unravel_index(indices, shape), dtype=np.int32)
    return tuple(ret)


unravel_index.support_native_out = False


def quantile(
    a: np.ndarray,
    q: Union[float, np.ndarray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    # quantile method in numpy backend, always return an array with dtype=float64.
    # in other backends, the output is the same dtype as the input.

    tuple(axis) if isinstance(axis, list) else axis

    return np.quantile(
        a, q, axis=axis, method=interpolation, keepdims=keepdims, out=out
    ).astype(a.dtype)


def corrcoef(
    x: np.ndarray,
    /,
    *,
    y: Optional[np.ndarray] = None,
    rowvar: Optional[bool] = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.corrcoef(x, y=y, rowvar=rowvar, dtype=x.dtype)


def nanmedian(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


nanmedian.support_native_out = True


def bincount(
    x: np.ndarray,
    /,
    *,
    weights: Optional[np.ndarray] = None,
    minlength: Optional[int] = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if weights is not None:
        ret = np.bincount(x, weights=weights, minlength=minlength)
        ret = ret.astype(weights.dtype)
    else:
        ret = np.bincount(x, minlength=minlength)
        ret = ret.astype(x.dtype)
    return ret


bincount.support_native_out = False


def histogram(data: np.ndarray, 
              input: Optional[Union[int, str, np.ndarray]] = None,
              name: Optional[np.ndarray[int]] = None, 
              weight: Union[bool, np.ndarray] = True,
              normed: Optional[bool] = None,
              step: Optional[float] = None,
              buckets: Optional[np.ndarray] = None,
              description: Optional[Union[np.ndarray[str], np.ndarray[float]]] = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the histogram of a dataset using NumPy.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    input : {None, int, str, np.ndarray}, optional
        This parameter controls the type of input data. If None (default), assumes that the input is already a 1D array. If an integer is given, assumes that the input is a 2D array with the specified number of columns. If a string is given, assumes that the input is a CSV file with the specified filename. If an array is given, assumes that it is a list of bins for the histogram.
    name : np.ndarray[int], optional
        An array of bin labels.
    weight : {bool, np.ndarray}, optional
        Whether to weight the input data. If True (default), each data point is weighted by 1.0. If a 1D array of weights is given, each data point is weighted by the corresponding weight.
    normed : bool, optional
        Whether to normalize the histogram. If True, the histogram will be normalized such that the integral over all bins is 1.0.
    step : float, optional
        The width of each bin in the histogram. If None (default), the bin width is automatically computed based on the data range and the number of bins.
    buckets : np.ndarray, optional
        An array of bin edges for the histogram. If None (default), the bin edges are automatically computed based on the data range and the number of bins.
    description : {np.ndarray[str], np.ndarray[float]}, optional
        A description of the histogram or a list of bin labels.

    Returns
    -------
    hist : np.ndarray
        The values of the histogram bins.
    bin_edges : np.ndarray
        The edges of the histogram bins, including the rightmost edge of the last bin.

    """
    # Determine the type of input data
    if input is None:
        x = data
    elif isinstance(input, int):
        x = data[:, input]
    elif isinstance(input, str):
        x = np.loadtxt(input, delimiter=',')
    elif isinstance(input, np.ndarray):
        x = data
    else:
        raise ValueError("Invalid value for parameter 'input'.")
    
    # Determine the bin edges
    if buckets is None:
        if step is None:
            step = (x.max() - x.min()) / 10.0
        bins = np.arange(x.min(), x.max() + step, step)
    else:
        bins = buckets
    
    # Compute the histogram
    hist, bin_edges = np.histogram(x, bins=bins, weights=weight, density=normed)
    
    return hist, bin_edges
