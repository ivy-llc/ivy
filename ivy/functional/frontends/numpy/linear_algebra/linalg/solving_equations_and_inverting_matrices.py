# global
import ivy

# inv


def inv(a):
    return ivy.inv(a)


inv.unsupported_dtypes = {"torch": ("float16",)}

# pinv


def pinv(a, rtol=1e-15, hermitian=False):
    return ivy.pinv(a, rtol)


pinv.unsupported_dtypes = ("float16",)
