from typing import Union, Optional, Sequence
import mxnet as mx


import ivy


def dirichlet(
    alpha: Union[(None, mx.ndarray.NDArray, float, Sequence[float])],
    /,
    *,
    size: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    seed: Optional[int] = None,
    dtype: Optional[None] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.dirichlet Not Implemented")


def beta(
    alpha: Union[(float, None, mx.ndarray.NDArray)],
    beta: Union[(float, None, mx.ndarray.NDArray)],
    /,
    *,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: Optional[str] = None,
    dtype: Optional[Union[(None, ivy.Dtype)]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.beta Not Implemented")


def gamma(
    alpha: Union[(float, None, mx.ndarray.NDArray)],
    beta: Union[(float, None, mx.ndarray.NDArray)],
    /,
    *,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: Optional[str] = None,
    dtype: Optional[Union[(None, ivy.Dtype)]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.gamma Not Implemented")


def poisson(
    lam: Union[(float, None, mx.ndarray.NDArray)],
    *,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: str,
    dtype: None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.poisson Not Implemented")


def bernoulli(
    probs: Union[(float, None, mx.ndarray.NDArray)],
    *,
    logits: Union[(float, None, mx.ndarray.NDArray)] = None,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: str,
    dtype: None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.bernoulli Not Implemented")
