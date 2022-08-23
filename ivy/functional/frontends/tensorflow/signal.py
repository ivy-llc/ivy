# global
import ivy


def ifftshift(x, axes=None, name=None):
    return ivy.ifftshift(x)


ifftshift.unsupported_dtypes = ("afloat16", "bfloat16")
