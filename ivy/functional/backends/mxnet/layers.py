"Collection of TensorFlow network layers, wrapped to fit Ivy syntax and signature."
from typing import Optional, Tuple, Union, Sequence
import ivy


def conv1d(
    x: Union[(None, tf.Variable)],
    filters: Union[(None, tf.Variable)],
    strides: Union[(int, Tuple[int])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    data_format: str = "NWC",
    dilations: Union[(int, Tuple[int])] = 1,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.conv1d Not Implemented")


def conv1d_transpose(
    x: Union[(None, tf.Variable)],
    filters: Union[(None, tf.Variable)],
    strides: Union[(int, Tuple[int])],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    data_format: str = "NWC",
    dilations: Union[(int, Tuple[int])] = 1,
    out: Optional[Union[(None, tf.Variable)]] = None,
):
    raise NotImplementedError("mxnet.conv1d_transpose Not Implemented")


def conv2d(
    x: Union[(None, tf.Variable)],
    filters: Union[(None, tf.Variable)],
    strides: Union[(int, Tuple[(int, int)])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[(int, Tuple[(int, int)])] = 1,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.conv2d Not Implemented")


def conv2d_transpose(
    x: Union[(None, tf.Variable)],
    filters: Union[(None, tf.Variable)],
    strides: Union[(int, Tuple[(int, int)])],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    data_format: str = "NHWC",
    dilations: Union[(int, Tuple[(int, int)])] = 1,
    out: Optional[Union[(None, tf.Variable)]] = None,
):
    raise NotImplementedError("mxnet.conv2d_transpose Not Implemented")


def depthwise_conv2d(
    x: Union[(None, tf.Variable)],
    filters: Union[(None, tf.Variable)],
    strides: Union[(int, Tuple[(int, int)])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[(int, Tuple[(int, int)])] = 1,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.depthwise_conv2d Not Implemented")


def conv3d(
    x: Union[(None, tf.Variable)],
    filters: Union[(None, tf.Variable)],
    strides: Union[(int, Tuple[(int, int, int)])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    data_format: str = "NDHWC",
    dilations: Union[(int, Tuple[(int, int, int)])] = 1,
    out: Optional[Union[(None, tf.Variable)]] = None,
):
    raise NotImplementedError("mxnet.conv3d Not Implemented")


def conv3d_transpose(
    x: None,
    filters: None,
    strides: Union[(int, Tuple[(int, int, int)])],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    data_format: str = "NDHWC",
    dilations: Union[(int, Tuple[(int, int, int)])] = 1,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> None:
    raise NotImplementedError("mxnet.conv3d_transpose Not Implemented")


def conv_general_dilated(
    x: Union[(None, tf.Variable)],
    filters: Union[(None, tf.Variable)],
    strides: Union[(int, Tuple[int], Tuple[(int, int)], Tuple[(int, int, int)])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[
        (int, Tuple[int], Tuple[(int, int)], Tuple[(int, int, int)])
    ] = 1,
    dilations: Union[(int, Tuple[int], Tuple[(int, int)], Tuple[(int, int, int)])] = 1,
    bias: Optional[Union[(None, tf.Variable)]] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.conv_general_dilated Not Implemented")


def conv_general_transpose(
    x: Union[(None, tf.Variable)],
    filters: Union[(None, tf.Variable)],
    strides: Union[(int, Tuple[(int, int)])],
    padding: str,
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    output_shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    dilations: Union[(int, Tuple[int], Tuple[(int, int)], Tuple[(int, int, int)])] = 1,
    feature_group_count: int = 1,
    bias: Optional[Union[(None, tf.Variable)]] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.conv_general_transpose Not Implemented")
