from numbers import Number
from typing import Optional, Union, Tuple
import ivy


def argmax(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[(ivy.Dtype, ivy.NativeDtype)]] = None,
    select_last_index: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.argmax Not Implemented")


def argmin(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[tf.dtypes.DType] = None,
    select_last_index: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.argmin Not Implemented")


def nonzero(
    x: Union[(None, tf.Variable)],
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[(None, tf.Variable, Tuple[Union[(None, tf.Variable)]])]:
    raise NotImplementedError("mxnet.nonzero Not Implemented")


def where(
    condition: Union[(None, tf.Variable)],
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.where Not Implemented")


def argwhere(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.argwhere Not Implemented")
