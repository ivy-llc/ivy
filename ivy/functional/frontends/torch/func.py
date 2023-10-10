import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_native_arrays,
)
from ivy.func_wrapper import outputs_to_ivy_arrays


def vmap(func, in_dims=0, out_dims=0, randomness="error", *, chunk_size=None):
    fun = outputs_to_native_arrays(func)
    return to_ivy_arrays_and_back(
        outputs_to_ivy_arrays(ivy.vmap(fun, in_axes=in_dims, out_axes=out_dims))
    )
