# local
import ivy.functional.frontends.numpy as np_frontend


def asmatrix(data, dtype=None):
    return np_frontend.matrix(data, dtype=dtype, copy=False)
