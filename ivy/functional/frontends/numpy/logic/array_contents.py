# global
import ivy


def isneginf(x, out=None):
    isinf = ivy.isinf(x)
    neg_sign_bit = ivy.less(x, 0)
    return ivy.logical_and(isinf, neg_sign_bit, out=out)


def isposinf(x, out=None):
    isinf = ivy.isinf(x)
    pos_sign_bit = ivy.bitwise_invert(ivy.less(x, 0))
    return ivy.logical_and(isinf, pos_sign_bit, out=out)
