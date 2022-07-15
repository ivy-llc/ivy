# global
import jax
from typing import Callable, Union, Sequence, Optional
# local
import ivy


def vmap(func: Callable,
         in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
         out_axes: Optional[int] = 0) -> Callable:
    return ivy.to_native_arrays_and_back(jax.vmap(func,
                                                  in_axes=in_axes,
                                                  out_axes=out_axes))
