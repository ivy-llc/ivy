# local

import ivy

from ivy.functional.frontends.numpy.func_wrapper import inputs_to_ivy_arrays
from ivy.functional.frontends.numpy import matrix

@inputs_to_ivy_arrays
def asmatrix(data, dtype=None):
    return ivy.matrix(data, dtype=dtype, copy=False)
    




