from typing import Union, Optional


def msort(
    a: Union[(None, tf.Variable, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.msort Not Implemented")


def lexsort(
    keys: Union[(None, tf.Variable)],
    /,
    *,
    axis: int = (-1),
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.lexsort Not Implemented")
