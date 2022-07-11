# global
import jax

# local
import  ivy


def vmap(func,
         in_axes=0,
         out_axes=0):
    return ivy.to_native_arrays_and_back(jax.vmap(func,
                                                  in_axes=in_axes,
                                                  out_axes=out_axes))
