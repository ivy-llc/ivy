import tensorflow

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_check_gather_nd_input_valid
from .tensorflow__helpers import tensorflow_gather_nd_helper
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_gather_nd(
    params: Union[tensorflow.Tensor, tensorflow.Variable],
    indices: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    batch_dims: int = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_gather_nd_input_valid(params, indices, batch_dims)
    try:
        return tensorflow.gather_nd(params, indices, batch_dims=batch_dims)
    except Exception:
        batch_dims %= len(params.shape)
        result = []
        if batch_dims == 0:
            result = tensorflow_gather_nd_helper(params, indices)
        else:
            for b in range(batch_dims):
                if b == 0:
                    zip_list = list(zip(params, indices))
                else:
                    zip_list = [
                        (p, i)
                        for z in [zip(p1, i1) for p1, i1 in zip_list]
                        for p, i in z
                    ]
            for z in zip_list:
                p, i = z[0], z[1]
                r = tensorflow_gather_nd_helper(p, i)
                result.append(r)
            result = tensorflow.stack(result)
            result = tensorflow.reshape(
                result,
                tensorflow.concat([params.shape[0:batch_dims], result.shape[1:]], 0),
            )
        return result
