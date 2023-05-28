import ivy

# in ivy/functional/frontends/tensorflow/nn.py
@to_ivy_array_and_back
def conv2D(input, weight, bias, stride, padding, data_format, dilations, name):
  return ivy.conv2D(weight, bias=bias, stride=stride, padding=padding, data_format=data_format, dilations=dilations, name=name)
