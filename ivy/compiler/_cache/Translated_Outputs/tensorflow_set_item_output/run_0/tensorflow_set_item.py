import tensorflow
import tensorflow as tf
import numpy as np

from typing import Tuple
from typing import Union
from typing import Optional

from .tensorflow__helpers import tensorflow__parse_query
from .tensorflow__helpers import tensorflow_asarray
from .tensorflow__helpers import tensorflow_astype_1
from .tensorflow__helpers import tensorflow_copy_array
from .tensorflow__helpers import tensorflow_handle_set_item
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_is_bool_dtype
from .tensorflow__helpers import tensorflow_nonzero
from .tensorflow__helpers import tensorflow_scatter_nd
from .tensorflow__helpers import tensorflow_set_item
from .tensorflow__helpers import tensorflow_shape
from .tensorflow__helpers import tensorflow_tile


@tensorflow_handle_set_item
def tensorflow_set_item(
    x: Union[tensorflow.Tensor, tf.Tensor],
    query: Union[tensorflow.Tensor, tf.Tensor, Tuple],
    val: Union[tensorflow.Tensor, tf.Tensor],
    /,
    *,
    copy: Optional[bool] = False,
):
    if isinstance(query, (list, tuple)) and any(
        [(q is Ellipsis or isinstance(q, slice) and q.stop is None) for q in query]
    ):
        np_array = x.numpy()
        np_array = tensorflow_set_item(np_array, query, np.asarray(val))
        return tensorflow_asarray(np_array)
    if copy:
        x = tensorflow_copy_array(x)
    if not tensorflow_is_array(val):
        val = tensorflow_asarray(val)
    if 0 in x.shape or 0 in val.shape:
        return x
    if tensorflow_is_array(query) and tensorflow_is_bool_dtype(query):
        if not len(query.shape):
            query = tensorflow_tile(query, (x.shape[0],))
        indices = tensorflow_nonzero(query, as_tuple=False)
    else:
        indices, target_shape, _ = tensorflow__parse_query(
            query, tensorflow_shape(x, as_array=True), scatter=True
        )
        if indices is None:
            return x
    val = tensorflow_astype_1(val, x.dtype)
    ret = tensorflow_scatter_nd(indices, val, reduction="replace", out=x)
    return ret
