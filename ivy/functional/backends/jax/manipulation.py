# global
import math
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List

# local
from ivy.functional.backends.jax import JaxArray


def squeeze(x: JaxArray,
            axis: Union[int, Tuple[int], List[int]]=None)\
        -> JaxArray:

        if x.shape == ():
            if axis is None or axis == 0 or axis == -1:
                return x
            raise ValueError('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
        return jnp.squeeze(x, axis)


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x

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


def permute_dims(x: JaxArray,
                axes: Tuple[int,...]) \
        -> JaxArray:
    return jnp.transpose(x,axes)


stack = jnp.stack
reshape = jnp.reshape

def concatenate(xs, axis=-1):
    if xs[0].shape == ():
        return jnp.concatenate([jnp.expand_dims(x, 0) for x in xs], axis)
    return jnp.concatenate(xs, axis)



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
clip = jnp.clip
constant_pad = lambda x, pad_width, value=0: jnp.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=value)
zero_pad = lambda x, pad_width: jnp.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=0)
swapaxes = jnp.swapaxes