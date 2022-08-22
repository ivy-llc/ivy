import ivy


def matrix_min(x, axis=None, f=None, keepdims=False):
    if len(x) == 0:
        return 0
    try:
        res = ivy.reduce_min(x, axis=axis, keepdims=keepdims, f=f)
        return res
    except:
        return "An exception occurred"
