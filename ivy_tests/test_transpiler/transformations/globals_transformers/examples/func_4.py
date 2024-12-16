import ivy


def ivy_function(x):
    res = ivy.sin(x) + ivy.pi
    lock = ivy.locks
    return res.dtype in ivy.valid_dtypes
