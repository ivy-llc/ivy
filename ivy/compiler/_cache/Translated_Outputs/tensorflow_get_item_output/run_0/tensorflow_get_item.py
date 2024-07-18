import tensorflow

from typing import Union
from typing import Optional
from typing import Tuple

from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_is_bool_dtype


def tensorflow_get_item(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    query: Union[tensorflow.Tensor, tensorflow.Variable, Tuple],
    *,
    copy: Optional[bool] = None,
):
    if (
        tensorflow_is_array(query)
        and tensorflow_is_bool_dtype(query)
        and not len(query.shape)
    ):
        return tensorflow.expand_dims(x, 0)
    return x[query]
