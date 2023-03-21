# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def angle(z, deg=False):
    angle = ivy.angle(z, deg=deg)
    if deg and len(z.shape) == 0:
        angle = ivy.astype(angle, ivy.float64)
    return angle


@to_ivy_arrays_and_back
def _imag(val):
    return ivy.imag(val)


@to_ivy_arrays_and_back
def _real(val):
    return ivy.real(val)
