# global
import ivy

# local
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back
)


@to_ivy_arrays_and_back
def nextafter(x1, x2, /, *, out=None):
    return ivy.nextafter(x1, x2, out=out)
