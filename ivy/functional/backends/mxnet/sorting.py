from typing import Union, Optional, Literal, List
import ivy


def argsort(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: int = (-1),
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.argsort Not Implemented")


def sort(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: int = (-1),
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.sort Not Implemented")


def searchsorted(
    x: Union[(None, tf.Variable)],
    v: Union[(None, tf.Variable)],
    /,
    *,
    side: Literal[("left", "right")] = "left",
    sorter: Optional[Union[(ivy.Array, ivy.NativeArray, List[int])]] = None,
    ret_dtype: None = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.searchsorted Not Implemented")
