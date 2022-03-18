# global
import math
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List

# local
from ivy.functional.backends.jax import JaxArray


# noinspection PyShadowingBuiltins
def flip(x: JaxArray,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
         -> JaxArray:
    return jnp.flip(x, axis=axis)


def expand_dims(x: JaxArray,
                axis: Optional[Union[int, Tuple[int], List[int]]] = None) \
        -> JaxArray:
    try:
        return jnp.expand_dims(x, axis)
    except ValueError as error:
        raise IndexError(error)


# Extra #
# ------#


def split(x, num_or_size_splits=None, axis=0, with_remainder=False):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception('input array had no shape, but num_sections specified was {}'.format(num_or_size_splits))
        return [x]
    if num_or_size_splits is None:
        num_or_size_splits = x.shape[axis]
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits] * num_chunks_int + [int(remainder * num_or_size_splits)]
    if isinstance(num_or_size_splits, (list, tuple)):
        num_or_size_splits = jnp.cumsum(jnp.array(num_or_size_splits[:-1]))
    return jnp.split(x, num_or_size_splits, axis)


repeat = jnp.repeat
tile = jnp.tile
