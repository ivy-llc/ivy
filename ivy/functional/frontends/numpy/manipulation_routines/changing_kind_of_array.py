# local
from ivy.functional.frontends.numpy.func_wrapper import inputs_to_ivy_arrays
import ivy.functional.frontends.numpy as np_frontend


@inputs_to_ivy_arrays
def asmatrix(data, dtype=None):
    return np_frontend.matrix(data, dtype=dtype, copy=False)
