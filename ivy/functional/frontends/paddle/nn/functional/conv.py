import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def conv2D(input, weight, bias, stride, padding, dilation, groups):
  return pdf.conv2D(weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
