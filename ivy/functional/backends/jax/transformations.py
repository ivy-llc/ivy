# global
import jax

# local
import  ivy


def vmap(fun,
         in_axes=0,
         out_axes=0,
         axis_name=None,
         axis_size=None):
    return ivy.to_native_arrays_and_back(jax.vmap(fun,
                                                  in_axes=in_axes,
                                                  out_axes=out_axes,
                                                  axis_name=axis_name,
                                                  axis_size=axis_size))
