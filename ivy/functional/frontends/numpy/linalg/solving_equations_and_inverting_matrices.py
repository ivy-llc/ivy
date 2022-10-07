# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


# solve
@to_ivy_arrays_and_back
def solve(a, b):
    return ivy.solve(a, b)


solve.unsupported_dtypes = ("float16",)


# inv
@to_ivy_arrays_and_back
def inv(a):
    return ivy.inv(a)


# pinv
@to_ivy_arrays_and_back
def pinv(a, rtol=1e-15, hermitian=False):
    return ivy.pinv(a, rtol)
