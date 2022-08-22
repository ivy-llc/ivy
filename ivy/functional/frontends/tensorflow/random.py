# global

import ivy
from typing import Union

def shuffle(value, seed: int = 0, name = None) -> ivy.Array:

    ivy.seed(seed_value = seed)

    return ivy.shuffle(value, out = ivy.array(name))


def normal(mean: Union[float, ivy.NativeArray, ivy.Array] = 0.0,
           stddev: Union[float, ivy.NativeArray, ivy.Array] = 1.0,
           shape: Union[float, ivy.NativeArray, ivy.Array] = None,
           dtype = None,
           seed = None,
           name='None') -> ivy.Array:

    if seed is not None:
        ivy.seed(seed_value = seed)

    return ivy.random_normal(mean = mean,
                             std = stddev,
                             shape = shape,
                             dtype = dtype,
                             device = None,
                             out = None)