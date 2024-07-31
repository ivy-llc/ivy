import tensorflow
import tensorflow as tf

from typing import Sequence
from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow__reshape_fortran_tf
from .tensorflow__helpers import tensorflow_check_elem_in_list
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_reshape(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            (new_s if con else old_s)
            for new_s, con, old_s in zip(
                shape, tensorflow.constant(shape) != 0, x.shape
            )
        ]
    if order == "F":
        return tensorflow__reshape_fortran_tf(x, shape)
    return tensorflow.reshape(x, shape)
