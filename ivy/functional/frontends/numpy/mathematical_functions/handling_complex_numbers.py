# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


# TODO working for complex input only, could be expanded to handle whole
# range of input which numpy is capable of taking
@to_ivy_arrays_and_back
def angle(z, deg=False):
    angle = ivy.angle(z, deg=deg)

    # numpy implementation multiplies by float64 when converting to degrees
    # so implicit casting occures, but only to single value inputs (not arrays)
    if deg and len(z.shape) == 0:
        angle = ivy.astype(angle, ivy.float64)
    return angle


@to_ivy_arrays_and_back
def _imag(val):
    return ivy.imag(val)


@to_ivy_arrays_and_back
def _real(val):
    return ivy.real(val)
