from typing import Union, Optional, Tuple, Sequence
import mxnet as mx


def histogram(
    a: None,
    /,
    *,
    bins: Optional[Union[(int, None, str)]] = None,
    axis: Optional[None] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[None] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[None] = None,
    density: Optional[bool] = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[None]:
    raise NotImplementedError("mxnet.histogram Not Implemented")


def median(
    input: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(Tuple[int], int)]] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.median Not Implemented")


def nanmean(
    a: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(int, Tuple[int])]] = None,
    keepdims: bool = False,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.nanmean Not Implemented")


def quantile(
    a: Union[(None, mx.ndarray.NDArray)],
    q: Union[(None, float)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    interpolation: str = "linear",
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.quantile Not Implemented")


def corrcoef(
    x: None,
    /,
    *,
    y: None,
    rowvar: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> None:
    raise NotImplementedError("mxnet.corrcoef Not Implemented")


def nanmedian(
    input: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(Tuple[int], int)]] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.nanmedian Not Implemented")


def bincount(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    weights: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    minlength: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.bincount Not Implemented")
