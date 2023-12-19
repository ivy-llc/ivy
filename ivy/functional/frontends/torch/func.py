import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_native_arrays,
)
from ivy.func_wrapper import outputs_to_ivy_arrays


def vmap(func, in_dims=0, out_dims=0, randomness="error", *, chunk_size=None):
    # Wrap the input function `func` to handle native arrays
    native_arrays_func = outputs_to_native_arrays(func)

    # Apply Ivy's vectorized map operation (ivy.vmap) with specified input and output dimensions
    ivy_vmap_result = ivy.vmap(native_arrays_func, in_axes=in_dims, out_axes=out_dims)

    # Wrap the result of Ivy's vmap operation to handle Ivy arrays
    ivy_arrays_result = to_ivy_arrays_and_back(outputs_to_ivy_arrays(ivy_vmap_result))

    return ivy_arrays_result
