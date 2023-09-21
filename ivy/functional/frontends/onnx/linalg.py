import ivy

from ivy.functional.frontends.onnx.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def MatMul(x1, x2):
    return ivy.matmul(x1, x2)
