# local

import ivy

from ivy.functional.frontends.numpy.func_wrapper import inputs_to_ivy_arrays

@inputs_to_ivy_arrays
def asmatrix(data, dtype=None):
    return ivy.matrix(data, dtype=dtype, copy=False)





