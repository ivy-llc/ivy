import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
def result_type(tensor, other):
    return ivy.result_type(tensor, other)


@to_ivy_arrays_and_back
def _assert(condition, message):
    if not condition:
        raise Exception(message)
    else:
        return True

@with_unsupported_dtypes({"1.11.0 and below": (
        "int8", "int16", "int32",
        'bool',
        'float16', 'float32', 'float64', 
        "complex64", 'complex38',
        'bfloat16',
        'uint8')}, "torch")
@to_ivy_arrays_and_back
def bincount(x, weights=None, minlength=0):
    return ivy.bincount(x, weights=weights, minlength=minlength)

