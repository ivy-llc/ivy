# local
import ivy
import ivy.functional.frontends as frontends
from ivy.func_wrapper import with_unsupported_dtypes

versions = frontends.versions["numpy"]


# solve
@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, versions)
def solve(a, b):
    return ivy.solve(a, b)


# inv
def inv(a):
    return ivy.inv(a)


# pinv
def pinv(a, rtol=1e-15, hermitian=False):
    return ivy.pinv(a, rtol)
