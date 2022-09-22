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
def tensorinv(a, b, reverse=False, ind=2):
    ret = ivy.tensorinv(a, )
    if reverse:
        return ivy.tensordot(ret, b, axes=2)
    return ret
