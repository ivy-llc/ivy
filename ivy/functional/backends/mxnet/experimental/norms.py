from typing import Union, Optional, Tuple


def l2_normalize(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[None] = None,
) -> None:
    raise NotImplementedError("mxnet.l2_normalize Not Implemented")


def batch_norm(
    x: Union[(None, tf.Variable)],
    mean: Union[(None, tf.Variable)],
    variance: Union[(None, tf.Variable)],
    /,
    *,
    scale: Optional[Union[(None, tf.Variable)]] = None,
    offset: Optional[Union[(None, tf.Variable)]] = None,
    training: bool = False,
    eps: float = 1e-05,
    momentum: float = 0.1,
    out: Optional[None] = None,
) -> Tuple[
    (Union[(None, tf.Variable)], Union[(None, tf.Variable)], Union[(None, tf.Variable)])
]:
    raise NotImplementedError("mxnet.batch_norm Not Implemented")


def instance_norm(
    x: Union[(None, tf.Variable)],
    mean: Union[(None, tf.Variable)],
    variance: Union[(None, tf.Variable)],
    /,
    *,
    scale: Optional[Union[(None, tf.Variable)]] = None,
    offset: Optional[Union[(None, tf.Variable)]] = None,
    training: bool = False,
    eps: float = 1e-05,
    momentum: float = 0.1,
    out: Optional[None] = None,
) -> Tuple[
    (Union[(None, tf.Variable)], Union[(None, tf.Variable)], Union[(None, tf.Variable)])
]:
    raise NotImplementedError("mxnet.instance_norm Not Implemented")


def lp_normalize(
    x: Union[(None, tf.Variable)],
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[None] = None,
) -> None:
    raise NotImplementedError("mxnet.lp_normalize Not Implemented")
