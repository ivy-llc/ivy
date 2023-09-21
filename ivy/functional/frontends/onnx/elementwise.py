import ivy

from ivy.functional.frontends.onnx.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def Abs(input):
    return ivy.abs(input)


@to_ivy_arrays_and_back
def Acos(input):
    return ivy.acos(input)


@to_ivy_arrays_and_back
def Acosh(input):
    return ivy.acosh(input)


@to_ivy_arrays_and_back
def Add(x1, x2):
    return ivy.add(x1, x2)


@to_ivy_arrays_and_back
def Asin(input):
    return ivy.asin(input)
