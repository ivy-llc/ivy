# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def _imag(val):
    return ivy.imag(val)


@to_ivy_arrays_and_back
def _angle(val,deg=False,) -> ivy.Array:
    return ivy.angle(val, deg=deg)
