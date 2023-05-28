import ivy
# in ivy/functional/frontends/tensorflow/nn.py
@to_ivy_arrays_and_back
def conv2D(input, weight, bias, stride, padding, dilation, groups):
  return pdf.conv2D(weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
