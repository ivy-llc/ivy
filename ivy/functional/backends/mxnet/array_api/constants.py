#global
import mxnet as mx
from typing import Union
#local
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out



@_handle_flat_arrays_in_out
def isnan(x: Union[mx.ndarray.ndarray.NDArray], str)-> []:
    return mx.nd.contrib.isnan(x).astype('bool')