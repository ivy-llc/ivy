# global
import functorch

# local
import ivy


def vmap(func, in_axis=0, out_axis=0):
    new_func = functorch.vmap(func, in_axis, out_axis)
    new_func = ivy.to_native_arrays_and_back(new_func)
    return new_func