import tensorflow

from typing import Optional

from .tensorflow__helpers import tensorflow__reshape_fortran_tf
from .tensorflow__helpers import tensorflow_check_elem_in_list
from .tensorflow__helpers import tensorflow_exists_bknd
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_inplace_update


@tensorflow_handle_array_like_without_promotion
def tensorflow_flatten(
    x: tensorflow.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    order: Optional[str] = "C",
    out: Optional[tensorflow.Tensor] = None,
):
    if x.shape == ():
        x = tensorflow.reshape(x, (1, -1))[0, :]
    if start_dim == end_dim:
        return tensorflow_inplace_update(out, x) if tensorflow_exists_bknd(out) else x
    if start_dim not in range(-x.shape.rank, x.shape.rank):
        raise IndexError(
            f"Dimension out of range (expected to be in range of {[-x.shape.rank, x.shape.rank - 1]}, but got {start_dim}"
        )
    if end_dim not in range(-x.shape.rank, x.shape.rank):
        raise IndexError(
            f"Dimension out of range (expected to be in range of {[-x.shape.rank, x.shape.rank - 1]}, but got {end_dim}"
        )
    if end_dim < 0:
        end_dim += x.shape.rank
    if start_dim < 0:
        start_dim += x.shape.rank
    if start_dim == end_dim:
        return x
    in_shape = tensorflow.shape(x)
    flattened_dim = tensorflow.math.reduce_prod(in_shape[start_dim : end_dim + 1])
    out_shape = tensorflow.concat(
        [in_shape[:start_dim], [flattened_dim], in_shape[end_dim + 1 :]], axis=0
    )
    tensorflow_check_elem_in_list(order, ["C", "F"])
    if order == "F":
        return tensorflow__reshape_fortran_tf(x, out_shape)
    return tensorflow.reshape(x, out_shape)
