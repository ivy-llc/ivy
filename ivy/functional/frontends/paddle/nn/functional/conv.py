# local
import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def conv2D(input, filters, stride=1, padding=0, data_format=None, dilations=None, out=None):
    return ivy.conv2D(input, filters, stride=stride, padding=padding, data_format=data_format, dilations=dilations, out=out)
