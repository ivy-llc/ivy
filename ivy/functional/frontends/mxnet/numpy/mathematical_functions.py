# local
import ivy
from ivy.functional.frontends.mxnet.func_wrapper import (
    to_ivy_arrays_and_back,
)
# from ivy.functional.frontends.mxnet.numpy import promote_types_of_mxnet_inputs


@to_ivy_arrays_and_back
def add(x1, x2):
    # x1, x2 = promote_types_of_mxnet_inputs(x1, x2)
    return ivy.add(x1, x2)
