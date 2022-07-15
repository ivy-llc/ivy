# global
import functorch
from typing import Callable, Union, Sequence, Optional

# local
import ivy


def vmap(func: Callable,
         in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
         out_axes: Optional[int] = 0) -> Callable:
    def _vmap(*args):
        new_fun = lambda *args: ivy.to_native(func(*args)) 
        new_func = functorch.vmap(new_fun, in_axes, out_axes)
        return ivy.to_ivy(new_func(*args))

    return _vmap
