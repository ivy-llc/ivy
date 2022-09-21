# local
import ivy


# solve
def solve(a, b):
    return ivy.solve(a, b)


solve.unsupported_dtypes = ("float16",)


# inv
def inv(a):
    return ivy.inv(a)


# pinv
def pinv(a, rtol=1e-15, hermitian=False):
    return ivy.pinv(a, rtol)


# tensorinv
def tensorinv(a, ind=2):
    return ivy.tensorinv(a, )
