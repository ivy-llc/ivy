import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def fftshift(x, axes=None, name=None):
    shape = x.shape

    if axes is None:
        axes = tuple(range(x.ndim))
        shifts = [(dim // 2) for dim in shape]
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = ivy.concat([shape[ax] // 2 for ax in axes])

    roll = ivy.roll(x, shifts, axis=axes)

    return roll
