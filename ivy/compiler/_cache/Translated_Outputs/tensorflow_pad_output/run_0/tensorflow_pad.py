import tensorflow

from typing import Tuple
from numbers import Number
from typing import Callable
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Literal
from typing import Union

from .tensorflow__helpers import tensorflow__to_tf_padding_bknd
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_pad(
    input: Union[tensorflow.Tensor, tensorflow.Variable],
    pad_width: Union[Iterable[Tuple[int]], int],
    /,
    *,
    mode: Union[
        Literal[
            "constant",
            "dilated",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
            "empty",
        ],
        Callable,
    ] = "constant",
    stat_length: Union[Iterable[Tuple[int]], int] = 1,
    constant_values: Union[Iterable[Tuple[Number]], Number] = 0,
    end_values: Union[Iterable[Tuple[Number]], Number] = 0,
    reflect_type: Literal["even", "odd"] = "even",
    **kwargs: Optional[Any],
):
    pad_width = tensorflow__to_tf_padding_bknd(pad_width, len(input.shape))
    if not isinstance(constant_values, (tensorflow.Variable, tensorflow.Tensor)):
        constant_values = tensorflow.constant(constant_values)
    if constant_values.dtype != input.dtype:
        constant_values = tensorflow.cast(constant_values, input.dtype)
    return tensorflow.pad(input, pad_width, mode=mode, constant_values=constant_values)
