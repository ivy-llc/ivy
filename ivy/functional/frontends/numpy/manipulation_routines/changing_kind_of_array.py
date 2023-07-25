# local
import ivy.functional.frontends.numpy as np_frontend
import ivy


def asmatrix(data, dtype=None):
    return np_frontend.matrix(ivy.array(data), dtype=dtype, copy=False)
