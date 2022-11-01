import ivy


def uniform(shape, minval=0, maxval=None, dtype=ivy.float32, seed=None, name=None):
    return ivy.random_uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, seed=seed
    )
