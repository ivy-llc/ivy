import ivy
import ivy.functional.frontends.paddle.nn.functional as pdf

def conv2D(input, weight, bias, stride, padding, data_format, dilations, name):
  return pdf.conv2D(weight, bias=bias, stride=stride, padding=padding, data_format=data_format, dilations=dilations, name=name)
