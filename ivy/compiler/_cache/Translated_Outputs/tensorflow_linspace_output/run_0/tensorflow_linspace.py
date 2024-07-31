import tensorflow

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow__slice_at_axis
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_infer_dtype


@tensorflow_infer_dtype
@tensorflow_handle_array_like_without_promotion
def tensorflow_linspace(
    start: Union[tensorflow.Tensor, tensorflow.Variable, float],
    stop: Union[tensorflow.Tensor, tensorflow.Variable, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: tensorflow.DType,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if axis is None:
        axis = -1
    start = tensorflow.cast(tensorflow.constant(start), dtype=dtype)
    stop = tensorflow.cast(tensorflow.constant(stop), dtype=dtype)
    if not endpoint:
        ans = tensorflow.linspace(start, stop, num + 1, axis=axis)
        if axis < 0:
            axis += len(ans.shape)
        ans = tensorflow.convert_to_tensor(
            ans.numpy()[tensorflow__slice_at_axis(slice(None, -1), axis)]
        )
    else:
        ans = tensorflow.linspace(start, stop, num, axis=axis)
    if dtype.is_integer and ans.dtype.is_floating:
        ans = tensorflow.math.floor(ans)
    return tensorflow.cast(ans, dtype)
