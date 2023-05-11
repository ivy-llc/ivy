import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_native_arrays,
)
from ivy.func_wrapper import outputs_to_ivy_arrays


def vmap(
    fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None, spmd_axis_name=None
):
    fun = outputs_to_native_arrays(fun)
    return to_ivy_arrays_and_back(
        outputs_to_ivy_arrays(ivy.vmap(fun, in_axes=in_axes, out_axes=out_axes))
    )


@to_ivy_arrays_and_back
def device_get(x):
    if ivy.dev(x) != "cpu":
        x = ivy.to_device(x, "cpu")
    return x


@to_ivy_arrays_and_back
def device_put(x, device=None, *, src=None):
    if device is not None:
        cur_dev = ivy.dev(x)
        device = ivy.as_ivy_dev(device)
        if cur_dev != device:
            x = ivy.to_device(x, device)
    return x
