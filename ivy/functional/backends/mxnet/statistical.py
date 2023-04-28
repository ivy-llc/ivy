from typing import Union, Optional, Sequence


def min(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.min Not Implemented")


def max(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.max Not Implemented")


def mean(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.mean Not Implemented")


def prod(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    dtype: Optional[None] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.prod Not Implemented")


def std(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    correction: Union[(int, float)] = 0.0,
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.std Not Implemented")


def sum(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    dtype: Optional[None] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.sum Not Implemented")


def var(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    correction: Union[(int, float)] = 0.0,
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.var Not Implemented")


def cumprod(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.cumprod Not Implemented")


def cumsum(
    x: Union[(None, tf.Variable)],
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.cumsum Not Implemented")


def einsum(
    equation: str,
    *operands: Union[(None, tf.Variable)],
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.einsum Not Implemented")
