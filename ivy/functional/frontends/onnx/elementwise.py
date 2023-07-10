import ivy

# import ivy.functional.frontends.onnx as onnx_frontend
from ivy.functional.frontends.onnx.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def abs(input):
    return ivy.abs(input)


@to_ivy_arrays_and_back
def acos(input):
    return ivy.acos(input)


@to_ivy_arrays_and_back
def acosh(input):
    return ivy.acosh(input)


@to_ivy_arrays_and_back
def add(x1, x2):
    return ivy.add(x1, x2)


@to_ivy_arrays_and_back
def asin(input):
    return ivy.asin(input)
