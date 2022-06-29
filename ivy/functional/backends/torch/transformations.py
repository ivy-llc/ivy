# global
import functorch

# local
import ivy


def vmap(func, in_axes=0, out_axes=0):
    new_func = functorch.vmap(func, in_axes, out_axes)
    new_func = ivy.to_native_arrays_and_back(new_func)
    return new_func