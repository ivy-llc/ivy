from typing import Union, Optional, Sequence
import ivy


def dirichlet(
    alpha: Union[(None, tf.Variable, float, Sequence[float])],
    /,
    *,
    size: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
    seed: Optional[int] = None,
    dtype: Optional[None] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.dirichlet Not Implemented")


def beta(
    alpha: Union[(float, None, tf.Variable)],
    beta: Union[(float, None, tf.Variable)],
    /,
    *,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: Optional[str] = None,
    dtype: Optional[Union[(None, ivy.Dtype)]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.beta Not Implemented")


def gamma(
    alpha: Union[(float, None, tf.Variable)],
    beta: Union[(float, None, tf.Variable)],
    /,
    *,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: Optional[str] = None,
    dtype: Optional[Union[(None, ivy.Dtype)]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.gamma Not Implemented")


def poisson(
    lam: Union[(float, None, tf.Variable)],
    *,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: str,
    dtype: None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.poisson Not Implemented")


def bernoulli(
    probs: Union[(float, None, tf.Variable)],
    *,
    logits: Union[(float, None, tf.Variable)] = None,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: str,
    dtype: None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.bernoulli Not Implemented")
