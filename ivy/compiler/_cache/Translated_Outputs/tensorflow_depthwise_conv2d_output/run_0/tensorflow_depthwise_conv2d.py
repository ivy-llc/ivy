import tensorflow

from typing import Union
from typing import Sequence
from typing import Optional
from typing import Tuple

from .tensorflow__helpers import tensorflow__extend_2d_padding
from .tensorflow__helpers import tensorflow_dev


def tensorflow_depthwise_conv2d(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    filters: Union[tensorflow.Tensor, tensorflow.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    permuted_x = False
    if data_format == "NCHW" and tensorflow_dev(x) == "cpu":
        x = tensorflow.transpose(x, (0, 2, 3, 1))
        data_format = "NHWC"
        permuted_x = True
    if tensorflow.rank(filters) == 3:
        filters = tensorflow.expand_dims(filters, -1)
    padding = tensorflow__extend_2d_padding(padding, data_format)
    strides = [1, strides[0], strides[1], 1]
    res = tensorflow.nn.depthwise_conv2d(
        x, filters, strides, padding, data_format, dilations
    )
    if permuted_x:
        res = tensorflow.transpose(res, (0, 3, 1, 2))
    return res
