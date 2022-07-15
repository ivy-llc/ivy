# global
import functorch

# local
import ivy


def vmap(func: Callable,
         in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
         out_axes: Optional[int] = 0) -> Callable:
    @ivy.to_native_arrays_and_back
    def _vmap(*args):
        new_func = functorch.vmap(func, in_axes, out_axes)
        return new_func(*args)
    return _vmap
