import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union
from typing import Tuple

from .tensorflow__helpers import tensorflow__parse_query_bknd
from .tensorflow__helpers import tensorflow_current_backend_str
from .tensorflow__helpers import tensorflow_gather_nd
from .tensorflow__helpers import tensorflow_is_array_bknd
from .tensorflow__helpers import tensorflow_is_bool_dtype_bknd
from .tensorflow__helpers import tensorflow_ndim_bknd_
from .tensorflow__helpers import tensorflow_nonzero
from .tensorflow__helpers import tensorflow_permute_dims
from .tensorflow__helpers import tensorflow_reshape
from .tensorflow__helpers import tensorflow_shape
from .tensorflow__helpers import tensorflow_zeros


def tensorflow_get_item_bknd(
    x: Union[tensorflow.Tensor, tf.Tensor],
    /,
    query: Union[tensorflow.Tensor, tf.Tensor, Tuple],
    *,
    copy: Optional[bool] = None,
):
    if tensorflow_is_array_bknd(query) and tensorflow_is_bool_dtype_bknd(query):
        if tensorflow_ndim_bknd_(query) == 0:
            if query is False:
                return tensorflow_zeros(shape=(0,) + x.shape, dtype=x.dtype)
            return x[None]
        query = tensorflow_nonzero(query, as_tuple=False)
        ret = tensorflow_gather_nd(x, query)
    else:
        x_shape = (
            x.shape
            if tensorflow_current_backend_str() == ""
            else tensorflow_shape(x, as_array=True)
        )
        query, target_shape, vector_inds = tensorflow__parse_query_bknd(query, x_shape)
        if vector_inds is not None:
            x = tensorflow_permute_dims(
                x,
                axes=[
                    *vector_inds,
                    *[i for i in range(len(x.shape)) if i not in vector_inds],
                ],
            )
        ret = tensorflow_gather_nd(x, query)
        ret = (
            tensorflow_reshape(ret, target_shape)
            if target_shape != list(ret.shape)
            else ret
        )
    return ret
