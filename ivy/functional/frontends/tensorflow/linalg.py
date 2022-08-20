import ivy
import tensorflow


def det(input, name=None):
    return ivy.det(input)


det.unsupported_dtypes = ("float16", "bfloat16")

def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)

eigvalsh.unsupported_dtypes = ("float16", "bfloat16")

