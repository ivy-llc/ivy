import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union
from typing import Sequence

from .tensorflow__helpers import tensorflow__broadcast_to_bknd
from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_check_equal
from .tensorflow__helpers import tensorflow_exists_bknd
from .tensorflow__helpers import tensorflow_gather_nd
from .tensorflow__helpers import tensorflow_inplace_update
from .tensorflow__helpers import tensorflow_multiply
from .tensorflow__helpers import tensorflow_promote_types_bknd


def tensorflow_scatter_nd(
    indices: Union[tensorflow.Tensor, tensorflow.Variable],
    updates: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Optional[Union[tf.TensorShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    updates_dtype = updates.dtype
    if tensorflow_exists_bknd(out):
        dtype = tensorflow_promote_types_bknd(out.dtype, updates_dtype)
    updates = tensorflow.cast(
        updates,
        (
            tensorflow_as_native_dtype(dtype)
            if tensorflow_exists_bknd(out)
            else updates_dtype
        ),
    )
    expected_shape = (
        list(tensorflow.shape(indices)[:-1])
        + list(out.shape[tensorflow.shape(indices)[-1] :])
        if tensorflow_exists_bknd(out)
        else list(tensorflow.shape(indices)[:-1])
        + list(shape[tensorflow.shape(indices)[-1] :])
    )
    updates = tensorflow__broadcast_to_bknd(updates, expected_shape)
    if len(updates.shape) == 0:
        indices = tensorflow.expand_dims(indices, 0)
        updates = tensorflow.expand_dims(updates, 0)
    target = out
    target_given = tensorflow_exists_bknd(target)
    if tensorflow_exists_bknd(shape) and target_given:
        tensorflow_check_equal(tuple(target.shape), tuple(shape), as_array=False)
    if not target_given:
        shape = list(shape) if tensorflow_exists_bknd(shape) else list(out.shape)
        target = tensorflow.zeros(shape, dtype=updates.dtype)
    if reduction == "sum":
        res = tensorflow.tensor_scatter_nd_add(target, indices, updates)
    elif reduction == "min":
        res = tensorflow.tensor_scatter_nd_min(target, indices, updates)
    elif reduction == "max":
        res = tensorflow.tensor_scatter_nd_max(target, indices, updates)
    elif reduction == "mul":
        updates = tensorflow_multiply(tensorflow_gather_nd(target, indices), updates)
        res = tensorflow.tensor_scatter_nd_update(target, indices, updates)
    elif reduction == "replace":
        res = tensorflow.tensor_scatter_nd_update(target, indices, updates)
    else:
        raise Exception(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max", "mul" or "replace"'
        )
    if tensorflow_exists_bknd(out):
        return tensorflow_inplace_update(out, res)
    return res
