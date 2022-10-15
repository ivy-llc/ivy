# local
import ivy


def normal(shape, mean=0.0, stddev=1.0, dtype=ivy.float32, seed=None, name=None):
    return ivy.random_normal(
        mean=mean, std=stddev, shape=ivy.shape(shape), dtype=dtype, seed=seed
    )
