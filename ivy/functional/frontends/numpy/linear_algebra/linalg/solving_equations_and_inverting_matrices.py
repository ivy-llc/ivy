# local
import ivy


# inv
def inv(a):
    return ivy.inv(a)


# pinv
def pinv(a, rtol=1e-15, hermitian=False):
    return ivy.pinv(a, rtol)
