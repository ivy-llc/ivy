# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def random_sample(size=None):
    return ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def dirichlet(alpha, size=None):
    return ivy.dirichlet(alpha, size=size)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def uniform(low=0.0, high=1.0, size=None):
    return ivy.random_uniform(low=low, high=high, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def geometric(p, size=None):
    if p < 0 or p > 1:
        raise ValueError("p must be in the interval [0, 1]")
    oneMinusP = ivy.subtract(1, p)
    sizeMinusOne = ivy.subtract(size, 1)

    return ivy.multiply(ivy.pow(oneMinusP, sizeMinusOne), p)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def normal(loc=0.0, scale=1.0, size=None):
    return ivy.random_normal(mean=loc, std=scale, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def poisson(lam=1.0, size=None):
    return ivy.poisson(lam=lam, shape=size)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def multinomial(n, pvals, size=None):
    assert not ivy.exists(size) or (len(size) > 0 and len(size) < 3)
    batch_size = 1
    if ivy.exists(size):
        if len(size) == 2:
            batch_size = size[0]
            num_samples = size[1]
        else:
            num_samples = size[0]
    else:
        num_samples = len(pvals)
    return ivy.multinomial(n, num_samples, batch_size=batch_size, probs=pvals)
