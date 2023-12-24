# local

import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)
from ivy import with_supported_dtypes


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def beta(a, b, size=None):
    return ivy.beta(a, b, shape=size)


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
def choice(a, size=None, replace=True, p=None):
    sc_size = 1
    if isinstance(size, int):
        sc_size = size
    elif size is not None:
        #  If the given shape is, e.g., (m, n, k)
        #  then m * n * k samples are drawn. As per numpy docs
        sc_size = 1
        for s in size:
            if s is not None:
                sc_size *= s
    if isinstance(a, int):
        a = ivy.arange(a)
    index = ivy.multinomial(len(a), sc_size, replace=replace, probs=p)
    return a[index]


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def dirichlet(alpha, size=None):
    return ivy.dirichlet(alpha, size=size)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def exponential(scale=1.0, size=None, dtype="float64"):
    if scale > 0:
        # Generate samples that are uniformly distributed based on given parameters
        u = ivy.random_uniform(low=0.0, high=0.0, shape=size, dtype=dtype)
        return ivy.exp(scale, out=u)
    return 0  # if scale parameter is less than or equal to 0


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def f(dfn, dfd, size=None):
    # Generate samples from the uniform distribution
    x1 = ivy.gamma(ivy.to_scalar(ivy.divide(dfn, 2)), 2.0, shape=size, dtype="float64")
    x2 = ivy.gamma(ivy.to_scalar(ivy.divide(dfd, 2)), 2.0, shape=size, dtype="float64")
    # Calculate the F-distributed samples
    samples = ivy.divide(ivy.divide(x1, ivy.array(dfn)), ivy.divide(x2, ivy.array(dfd)))
    return samples


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def gamma(shape, scale=1.0, size=None):
    return ivy.gamma(shape, scale, shape=size, dtype="float64")


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
def gumbel(loc=0.0, scale=1.0, size=None):
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    x = loc - scale * ivy.log(-ivy.log(u))
    return x


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def laplace(loc=0.0, scale=1.0, size=None):
    u = ivy.random_uniform(low=0.0, high=0.0, shape=size, dtype="float64")
    u = loc - scale * ivy.sign(u - 0.5) * ivy.log(1 - 2 * ivy.abs(u - 0.5))
    return u


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def logistic(loc=0.0, scale=1.0, size=None):
    u = ivy.random_uniform(low=0.0, high=0.0, shape=size, dtype="float64")
    x = loc + scale * ivy.log(u / (1 - u))
    return x


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def lognormal(mean=0.0, sigma=1.0, size=None):
    ret = ivy.exp(ivy.random_normal(mean=mean, std=sigma, shape=size, dtype="float64"))
    return ret


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def logseries(p=0, size=None):
    if p < 0 or p >= 1:
        raise ValueError("p value must be in the open interval (0, 1)")
    r = ivy.log(1 - p)
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size)
    v = ivy.random_uniform(low=0.0, high=1.0, shape=size)
    q = 1 - ivy.exp(r * u)
    ret = 1 + ivy.log(v) / ivy.log(q)
    return ret


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


@with_supported_dtypes(
    {"1.25.2 and below": ("float16", "float32")},
    "numpy",
)
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def noncentral_chisquare(df, nonc, size=None):
    if ivy.any(df <= 0):
        raise ValueError("Degree of freedom must be greater than 0")
    if ivy.has_nans(nonc):
        return ivy.nan
    if ivy.any(nonc == 0):
        return chisquare(df, size=size)
    if ivy.any(df < 1):
        n = standard_normal() + ivy.sqrt(nonc)
        return chisquare(df - 1, size=size) + n * n
    else:
        i = poisson(nonc / 2.0, size=size)
        return chisquare(df + 2 * i, size=size)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def normal(loc=0.0, scale=1.0, size=None):
    return ivy.random_normal(mean=loc, std=scale, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def pareto(a, size=None):
    if a < 0:
        return 0
    u = ivy.random_uniform(low=0.0, high=0.0, shape=size, dtype="float64")
    return ivy.pow(1 / (1 - u), 1 / a)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def permutation(x, /):
    if isinstance(x, int):
        x = ivy.arange(x)
    return ivy.shuffle(x)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def poisson(lam=1.0, size=None):
    return ivy.poisson(lam=lam, shape=size)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def random_sample(size=None):
    return ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def rayleigh(scale, size=None):
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    log_u = ivy.log(u)
    x = ivy.multiply(scale, ivy.sqrt(ivy.multiply(-2, log_u)))
    return x


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def shuffle(x, axis=0, /):
    if isinstance(x, int):
        x = ivy.arange(x)
    return ivy.shuffle(x, axis)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_cauchy(size=None):
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return ivy.tan(ivy.pi * (u - 0.5))


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_exponential(size=None):
    if size is None:
        size = 1
    U = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return -ivy.log(U)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_gamma(shape, size=None):
    return ivy.gamma(shape, 1.0, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_normal(size=None):
    return ivy.random_normal(mean=0.0, std=1.0, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_t(df, size=None):
    numerator = ivy.random_normal(mean=0.0, std=1.0, shape=size, dtype="float64")
    denominator = ivy.gamma(df / 2, 1.0, shape=size, dtype="float64")
    return ivy.sqrt(df / 2) * ivy.divide(numerator, ivy.sqrt(denominator))


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


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def uniform(low=0.0, high=1.0, size=None):
    return ivy.random_uniform(low=low, high=high, shape=size, dtype="float64")


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def vonmises(mu, kappa, size=None):
    t_size = 0
    # Output shape. If the given shape is, e.g., (m, n, k),
    # then m * n * k samples are drawn.
    if size is None or len(size) == 0:
        t_size = 1
    else:
        for x in size:
            t_size = t_size * x
    size = t_size
    li = []
    while len(li) < size:
        # Generate samples from the von Mises distribution using numpy
        u = ivy.random_uniform(low=-ivy.pi, high=ivy.pi, shape=size)
        v = ivy.random_uniform(low=0, high=1, shape=size)

        condition = v < (1 + ivy.exp(kappa * ivy.cos(u - mu))) / (
            2 * ivy.pi * ivy.i0(kappa)
        )
        selected_samples = u[condition]
        li.extend(ivy.to_list(selected_samples))

    return ivy.array(li[:size])


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def wald(mean, scale, size=None):
    if size is None:
        size = 1
    mu_2l = mean / (2 * scale)
    Y = ivy.random_normal(mean=0, std=1, shape=size, dtype="float64")
    U = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")

    Y = mean * ivy.square(Y)
    X = mean + mu_2l * (Y - ivy.sqrt(((4 * scale) * Y) + ivy.square(Y)))

    condition = mean / (mean + X) >= U
    value1 = X
    value2 = mean * mean / X

    return ivy.where(condition, value1, value2)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def weibull(a, size=None):
    if a < 0:
        return 0
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return ivy.pow(-ivy.log(1 - u), 1 / a)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def zipf(a, size=None):
    if a <= 1:
        return 0
    u = ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return ivy.floor(ivy.pow(1 / (1 - u), 1 / a))
