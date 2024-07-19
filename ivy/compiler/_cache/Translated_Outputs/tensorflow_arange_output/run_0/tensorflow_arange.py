import tensorflow

from typing import Union
from typing import Optional

from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_default_dtype_bknd
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[tensorflow.DType] = None,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if stop is None:
        stop = start
        start = 0
    if step > 0 and start > stop or step < 0 and start < stop:
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start
    if isinstance(start, (float, int)):
        start = tensorflow.convert_to_tensor(start)
    if isinstance(stop, (float, int)):
        stop = tensorflow.convert_to_tensor(stop)
    if isinstance(step, (float, int)):
        step = tensorflow.convert_to_tensor(step)
    if dtype is None:
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            return tensorflow.cast(
                tensorflow.range(start, stop, delta=step, dtype=tensorflow.int64),
                tensorflow.int32,
            )
        else:
            return tensorflow.range(start, stop, delta=step)
    else:
        dtype = tensorflow_as_native_dtype(tensorflow_default_dtype_bknd(dtype=dtype))
        if dtype in [
            tensorflow.int8,
            tensorflow.uint8,
            tensorflow.int16,
            tensorflow.uint16,
            tensorflow.uint32,
            tensorflow.uint64,
        ]:
            return tensorflow.cast(
                tensorflow.range(start, stop, delta=step, dtype=tensorflow.int64), dtype
            )
        else:
            return tensorflow.range(start, stop, delta=step, dtype=dtype)
