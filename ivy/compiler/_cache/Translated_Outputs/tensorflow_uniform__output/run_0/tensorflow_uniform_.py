from .tensorflow__helpers import tensorflow_astype
from .tensorflow__helpers import tensorflow_inplace_update
from .tensorflow__helpers import tensorflow_random_uniform


def tensorflow_uniform_(arr, from_=0, to=1, *, generator=None):
    ret = tensorflow_random_uniform(
        low=from_, high=to, shape=arr.shape, dtype=arr.dtype, seed=generator
    )
    arr = tensorflow_inplace_update(arr, tensorflow_astype(ret, arr.dtype))
    return arr
