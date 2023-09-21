# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


def asmatrix(data, dtype=None):
    return np_frontend.matrix(ivy.array(data), dtype=dtype, copy=False)


def asscalar(a):
    return a.item()
