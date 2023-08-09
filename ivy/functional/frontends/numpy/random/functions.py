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


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def permutation(x, /):
    if isinstance(x, int):
        x = ivy.arange(x)
    return ivy.shuffle(x)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def beta(a, b, size=None):
    return ivy.beta(a, b, shape=size)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def shuffle(x, axis=0, /):
    if isinstance(x, int):
        x = ivy.arange(x)
    return ivy.shuffle(x, axis)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_normal(size=None):
    return ivy.random_normal(mean=0.0, std=1.0, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_gamma(shape, size=None):
    return ivy.gamma(shape, 1.0, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_t(df, size=None):
    numerator = ivy.random_normal(mean=0.0, std=1.0, shape=size, dtype="float64")
    denominator = ivy.gamma(df / 2, 1.0, shape=size, dtype="float64")
    return ivy.sqrt(df / 2) * ivy.divide(numerator, ivy.sqrt(denominator))


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def binomial(n, p, size=None):
    if p < 0 or p > 1:
        raise ValueError("p must be in the interval (0, 1)")
    if n < 0:
        raise ValueError("n must be strictly positive")
    if size is None:
        size = 1
    else:
        size = size
    if isinstance(size, int):
        size = (size,)
    lambda_ = ivy.multiply(n, p)
    return ivy.poisson(lambda_, shape=size)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def chisquare(df, size=None):
    df = ivy.array(df)  # scalar ints and floats are also array_like
    if ivy.any(df <= 0):
        raise ValueError("df <= 0")

    # ivy.gamma() throws an error if both alpha is an array and a shape is passed
    # so this part broadcasts df into the shape of `size`` first to keep it happy.
    if size is not None:
        df = df * ivy.ones(size)

    return ivy.gamma(df / 2, 2, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def lognormal(mean=0.0, sigma=1.0, size=None):
    ret = ivy.exp(ivy.random_normal(mean=mean, std=sigma, shape=size, dtype="float64"))
    return ret


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def negative_binomial(n, p, size=None):
    if p <= 0 or p >= 1:
        raise ValueError("p must be in the interval (0, 1)")
    if n <= 0:
        raise ValueError("n must be strictly positive")
    # numpy implementation uses scale = (1 - p) / p
    scale = (1 - p) / p
    # poisson requires shape to be a tuple
    if isinstance(size, int):
        size = (size,)
    lambda_ = ivy.gamma(n, scale, shape=size)
    return ivy.poisson(lam=lambda_, shape=size)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def weibull(a, size=None):
    if a < 0:
        return 0
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return ivy.pow(-ivy.log(1 - u), 1 / a)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_cauchy(size=None):
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return ivy.tan(ivy.pi * (u - 0.5))


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def rayleigh(scale, size=None):
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    log_u = ivy.log(u)
    x = ivy.multiply(scale, ivy.sqrt(ivy.multiply(-2, log_u)))
    return x


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def gumbel(loc=0.0, scale=1.0, size=None):
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    x = loc - scale * ivy.log(-ivy.log(u))
    return x


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def gamma(shape, scale=1.0, size=None):
    return ivy.gamma(shape, scale, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def logistic(loc=0.0, scale=1.0, size=None):
    u = ivy.random_uniform(low=0.0, high=0.0, shape=size, dtype="float64")
    x = loc + scale * ivy.log(u / (1 - u))
    return x


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def triangular(left, mode, right, size=None):
    if left > mode or mode > right or left == right:
        raise ivy.utils.exceptions.IvyValueError(
            "left < mode < right is not being followed"
        )
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    condition = u <= (mode - left) / (right - left)
    values1 = left + (right - left) * (u * (mode - left) / (right - left)) ** 0.5
    values2 = (
        right - (right - mode) * ((1 - u) * (right - mode) / (right - left)) ** 0.5
    )
    return ivy.where(condition, values1, values2)
