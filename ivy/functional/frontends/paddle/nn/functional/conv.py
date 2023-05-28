import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def conv2D(input, weight, bias, stride, padding, data_format, dilations, name):
  return ivy.conv2D(weight, bias=bias, stride=stride, padding=padding, data_format=data_format, dilations=dilations, name=name)
