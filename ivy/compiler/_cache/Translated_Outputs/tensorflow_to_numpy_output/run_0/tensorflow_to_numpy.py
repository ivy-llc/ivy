import tensorflow
import numpy as np

from typing import Union

from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_get_num_dims
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_is_array_bknd


@tensorflow_handle_array_like_without_promotion
def tensorflow_to_numpy(
    x: Union[tensorflow.Tensor, tensorflow.Variable], /, *, copy: bool = True
):
    if (
        tensorflow_is_array_bknd(x)
        and tensorflow_get_num_dims(x) == 0
        and tensorflow_as_native_dtype(x.dtype) is tensorflow.bfloat16
    ):
        x = tensorflow.expand_dims(x, 0)
        if copy:
            return np.squeeze(np.array(tensorflow.convert_to_tensor(x)), 0)
        else:
            return np.squeeze(np.asarray(tensorflow.convert_to_tensor(x)), 0)
    if copy:
        return np.array(tensorflow.convert_to_tensor(x))
    else:
        return np.asarray(tensorflow.convert_to_tensor(x))
