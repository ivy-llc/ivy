# global
from typing import (
    Optional,
    Union,
    Sequence,
    Tuple,
    NamedTuple,
    Literal,
    Callable,
    Any,
    List,
)
from numbers import Number
import numpy as np

# local
import ivy
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array


def moveaxis(
    a: np.ndarray,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.moveaxis(a, source, destination)


moveaxis.support_native_out = False


def heaviside(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.heaviside(
        x1,
        x2,
        out=out,
    )


heaviside.support_native_out = True


def flipud(
    m: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.flipud(m)


flipud.support_native_out = False


def vstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.vstack(arrays)


def hstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.hstack(arrays)


def rot90(
    m: np.ndarray,
    /,
    *,
    k: Optional[int] = 1,
    axes: Optional[Tuple[int, int]] = (0, 1),
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.rot90(m, k, axes)


def top_k(
    x: np.ndarray,
    k: int,
    /,
    *,
    axis: Optional[int] = -1,
    largest: Optional[bool] = True,
    out: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not largest:
        indices = np.argsort(x, axis=axis)
        indices = np.take(indices, np.arange(k), axis=axis)
    else:
        x = -x
        indices = np.argsort(x, axis=axis)
        indices = np.take(indices, np.arange(k), axis=axis)
        x = -x
    topk_res = NamedTuple("top_k", [("values", np.ndarray), ("indices", np.ndarray)])
    val = np.take_along_axis(x, indices, axis=axis)
    return topk_res(val, indices)


def fliplr(
    m: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.fliplr(m)


fliplr.support_native_out = False


def i0(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.i0(x)


i0.support_native_out = False


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def pad(
    input: np.ndarray,
    pad_width: Union[Sequence[Sequence[int]], np.ndarray, int],
    /,
    *,
    mode: Optional[
        Union[
            Literal[
                "constant",
                "edge",
                "linear_ramp",
                "maximum",
                "mean",
                "median",
                "minimum",
                "reflect",
                "symmetric",
                "wrap",
                "empty",
            ],
            Callable,
        ]
    ] = "constant",
    stat_length: Optional[Union[Sequence[Sequence[int]], int]] = None,
    constant_values: Optional[Union[Sequence[Sequence[Number]], Number]] = 0,
    end_values: Optional[Union[Sequence[Sequence[Number]], Number]] = 0,
    reflect_type: Optional[Literal["even", "odd"]] = "even",
    **kwargs: Optional[Any],
) -> np.ndarray:
    if callable(mode):
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            **kwargs,
        )
    if mode in ["maximum", "mean", "median", "minimum"]:
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            stat_length=stat_length,
        )
    elif mode == "constant":
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            constant_values=constant_values,
        )
    elif mode == "linear_ramp":
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            end_values=end_values,
        )
    elif mode in ["reflect", "symmetric"]:
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            reflect_type=reflect_type,
        )
    else:
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
        )


def vsplit(
    ary: np.ndarray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[np.ndarray]:
    return np.vsplit(ary, indices_or_sections)


def dsplit(
    ary: np.ndarray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[np.ndarray]:
    if ary.ndim < 3:
        raise ivy.utils.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return np.dsplit(ary, indices_or_sections)


def atleast_1d(*arys: Union[np.ndarray, bool, Number]) -> List[np.ndarray]:
    return np.atleast_1d(*arys)


def dstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.dstack(arrays)


def atleast_2d(*arys: np.ndarray) -> List[np.ndarray]:
    return np.atleast_2d(*arys)


def atleast_3d(*arys: Union[np.ndarray, bool, Number]) -> List[np.ndarray]:
    return np.atleast_3d(*arys)


@_scalar_output_to_0d_array
def take_along_axis(
    arr: np.ndarray,
    indices: np.ndarray,
    axis: int,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if arr.ndim != indices.ndim and axis is not None:
        raise ivy.utils.exceptions.IvyException(
            "arr and indices must have the same number of dimensions;"
            + f" got {arr.ndim} vs {indices.ndim}"
        )
    return np.take_along_axis(arr, indices, axis)


def hsplit(
    ary: np.ndarray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[np.ndarray]:
    return np.hsplit(ary, indices_or_sections)


take_along_axis.support_native_out = False


def broadcast_shapes(shapes: Union[List[int], List[Tuple]]) -> List[int]:
    return np.broadcast_shapes(*shapes)


def expand(
    x: np.ndarray,
    shape: Union[List[int], List[Tuple]],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    shape = list(shape)
    for i, dim in enumerate(shape):
        if dim < 0:
            shape[i] = x.shape[i]
    return np.broadcast_to(x, tuple(shape))


expand.support_native_out = False


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