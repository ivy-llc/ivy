# global
import functorch

# local
import ivy


def vmap(func, in_axes=0, out_axes=0):
    @ivy.to_native_arrays_and_back
    def _vmap(*args, **kwargs):
        new_func = functorch.vmap(func, in_axes, out_axes)
        return new_func(*args, **kwargs)
    return _vmap