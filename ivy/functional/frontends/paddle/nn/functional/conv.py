import ivy
from ivy.functional.frontends.tensorflow.nn import conv2D as paddle_conv2D

def conv2D(input, weight, bias, stride, padding, data_format, dilations, name):
  return paddle_conv2D(weight, bias=bias, stride=stride, padding=padding, data_format=data_format, dilations=dilations, name=name)
