import ivy


def array(obj, dtype=None, copy=True, ndmin=4):
    ret = ivy.array(obj, dtype=dtype, copy=copy)
    while ndmin > len(ret.shape):
        ret = ivy.expand_dims(ret, axis=0)
    return ret
