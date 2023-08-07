# local
import ivy
from ivy.functional.frontends.mxnet.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_mxnet_out,
)
from ivy.functional.frontends.mxnet.numpy import promote_types_of_mxnet_inputs


@handle_mxnet_out
@to_ivy_arrays_and_back
def add(x1, x2, out=None):
    x1, x2 = promote_types_of_mxnet_inputs(x1, x2)
    return ivy.add(x1, x2, out=out)


@handle_mxnet_out
@to_ivy_arrays_and_back
def sin(x, out=None, **kwargs):
    return ivy.sin(x, out=out, **kwargs)
