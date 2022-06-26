#global
import ivy


def tan(x, name=None):
    # Todo: name argument
    return ivy.tan(x)

tan.unsupported_dtypes = {"torch": ("float16",)}
