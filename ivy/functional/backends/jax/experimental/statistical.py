from typing import Optional, Union, Tuple, Sequence
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp


def median(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return jnp.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )


def nanmean(
    a: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return jnp.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def unravel_index(
    indices: JaxArray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> Tuple:
    return jnp.unravel_index(indices.astype(jnp.int32), shape)


def quantile(
    a: JaxArray,
    q: Union[float, JaxArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    interpolation: Optional[str] = "linear",
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:

    if isinstance(axis, list):
        axis = tuple(axis)

    return jnp.quantile(
        a, q, axis=axis, method=interpolation, keepdims=keepdims, out=out
    )


def corrcoef(
    x: JaxArray,
    /,
    *,
    y: Optional[JaxArray] = None,
    rowvar: Optional[bool] = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.corrcoef(x, y=y, rowvar=rowvar)


def nanmedian(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


def bincount(
    x: JaxArray,
    /,
    *,
    weights: Optional[JaxArray] = None,
    minlength: Optional[int] = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if weights is not None:
        ret = jnp.bincount(x, weights=weights, minlength=minlength)
        ret = ret.astype(weights.dtype)
    else:
        ret = jnp.bincount(x, minlength=minlength).astype(x.dtype)
    return ret


import numpy as np


def jax_histogram(data: np.ndarray, 
                  input: Optional[Union[int, str, np.ndarray]] = None,
                  name: Optional[str] = None, 
                  weight: Optional[Union[bool, np.ndarray]] = True,
                  normed: Optional[bool] = None,
                  step: Optional[Union[float, int]] = None,
                  buckets: Optional[np.ndarray] = None,
                  description: Optional[Union[str, np.ndarray]] = None,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the histogram of a dataset using JAX.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    input : {None, int, str, np.ndarray}, optional
        This parameter controls the type of input data. If None (default), assumes that the input is already a 1D array. If an integer is given, assumes that the input is a 2D array with the specified number of columns. If a string is given, assumes that the input is a CSV file with the specified filename. If an array is given, assumes that it is a list of bins for the histogram.
    name : str, optional
        The name of the histogram. This parameter is ignored in this function.
    weight : {bool, np.ndarray}, optional
        Whether to weight the input data. If True (default), each data point is weighted by 1.0. If a 1D array of weights is given, each data point is weighted by the corresponding weight.
    normed : bool, optional
        Whether to normalize the histogram. If True, the histogram will be normalized such that the integral over all bins is 1.0.
    step : {float, int}, optional
        The width of each bin in the histogram. If None (default), the bin width is automatically computed based on the data range and the number of bins.
    buckets : np.ndarray, optional
        An array of bin edges for the histogram. If None (default), the bin edges are automatically computed based on the data range and the number of bins.
    description : {str, np.ndarray}, optional
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
