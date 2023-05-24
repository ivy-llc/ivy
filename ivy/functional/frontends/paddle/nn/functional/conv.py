import ivy
import ivy.functional.frontends.paddle.nn.functional as pdf

def conv2D(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
  return pdf.conv2D(weight, bias, stride, padding, dilation, groups)
