# global
import operator

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_jax_dtype,
)


@to_ivy_arrays_and_back
def PRNGKey(seed):
    return ivy.array([0, seed % 4294967295 - (seed // 4294967295)], dtype=ivy.int64)


@handle_jax_dtype
@to_ivy_arrays_and_back
def uniform(key, shape=(), dtype=None, minval=0.0, maxval=1.0):
    return ivy.random_uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, seed=ivy.to_scalar(key[1])
    )


@handle_jax_dtype
@to_ivy_arrays_and_back
def normal(key, shape=(), dtype=None):
    return ivy.random_normal(shape=shape, dtype=dtype, seed=ivy.to_scalar(key[1]))


def _get_seed(key):
    key1, key2 = int(key[0]), int(key[1])
    return ivy.to_scalar(int("".join(map(str, [key1, key2]))))


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def beta(key, a, b, shape=None, dtype=None):
    seed = _get_seed(key)
    return ivy.beta(a, b, shape=shape, dtype=dtype, seed=seed)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def dirichlet(key, alpha, shape=None, dtype="float32"):
    seed = _get_seed(key)
    alpha = ivy.astype(alpha, dtype)
    return ivy.dirichlet(alpha, size=shape, dtype=dtype, seed=seed)


@handle_jax_dtype
@to_ivy_arrays_and_back
def cauchy(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    u = ivy.random_uniform(low=0.0, high=1.0, shape=shape, dtype=dtype, seed=seed)
    return ivy.tan(ivy.pi * (u - 0.5))


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.13 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def poisson(key, lam, shape=None, dtype=None):
    seed = _get_seed(key)
    return ivy.poisson(lam, shape=shape, dtype=dtype, seed=seed, fill_value=-1)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def gamma(key, a, shape=None, dtype="float64"):
    seed = _get_seed(key)
    return ivy.gamma(a, 1.0, shape=shape, dtype=dtype, seed=seed)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def gumbel(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform_x = ivy.random_uniform(
        low=0.0,
        high=1.0,
        shape=shape,
        dtype=dtype,
        seed=seed,
    )
    return -ivy.log(-ivy.log(uniform_x))


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.13 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def rademacher(key, shape, dtype="int64"):
    seed = _get_seed(key)
    b = ivy.bernoulli(ivy.array([0.5]), shape=shape, dtype="float32", seed=seed)
    b = ivy.astype(b, dtype)
    return 2 * b - 1


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def generalized_normal(key, p, shape=(), dtype="float64"):
    seed = _get_seed(key)
    g = ivy.gamma(1 / p, 1.0, shape=shape, dtype=dtype, seed=seed)
    b = ivy.bernoulli(ivy.array([0.5]), shape=shape, dtype=dtype, seed=seed)
    r = 2 * b - 1
    return r * g ** (1 / p)


def t(key, df, shape=(), dtype="float64"):
    seed = _get_seed(key)
    n = ivy.random_normal(shape=shape, dtype=dtype, seed=seed)
    half_df = df / 2.0
    g = ivy.gamma(half_df, 1.0, shape=shape, dtype=dtype, seed=seed)
    return n * ivy.sqrt(ivy.divide(half_df, g))


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.13 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def randint(key, shape, minval, maxval, dtype="int64"):
    seed = _get_seed(key)
    return ivy.randint(minval, maxval, shape=shape, dtype=dtype, seed=seed)


@to_ivy_arrays_and_back
def bernoulli(key, p=0.5, shape=None):
    seed = _get_seed(key)
    return ivy.bernoulli(p, shape=shape, seed=seed)


@to_ivy_arrays_and_back
def fold_in(key, data):
    s = ivy.bitwise_left_shift(
        ivy.asarray(data, dtype=ivy.uint32), ivy.array(32, dtype=ivy.uint32)
    )
    return ivy.bitwise_xor(key, s)


@to_ivy_arrays_and_back
def permutation(key, x, axis=0, independent=False):
    x = ivy.array(x)
    seed = _get_seed(key)
    if not ivy.get_num_dims(x):
        r = int(x)
        return ivy.shuffle(ivy.arange(r), axis, seed=seed)
    if independent:
        return ivy.shuffle(x, axis, seed=seed)
    rand = ivy.arange(x.shape[axis])
    ind = ivy.shuffle(rand, 0, seed=seed)
    return ivy.gather(x, ind, axis=axis)


# loggamma
@to_ivy_arrays_and_back
@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def loggamma(key, a, shape=None, dtype="float64"):
    seed = _get_seed(key)
    return ivy.log(ivy.gamma(a, 1.0, shape=shape, dtype=dtype, seed=seed))


@to_ivy_arrays_and_back
def shuffle(key, x, axis=0):
    seed = _get_seed(key)
    x = ivy.flip(x, axis=axis)
    return ivy.shuffle(x, seed=seed)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def exponential(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform = ivy.random_uniform(seed=seed, shape=shape, dtype=dtype)
    exp = -ivy.log(1 - uniform)
    return exp


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def weibull_min(key, scale, concentration, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform_x = ivy.random_uniform(seed=seed, shape=shape, dtype=dtype)
    x = 1 - uniform_x
    weibull = x ** (concentration - 1) * -ivy.log(x / scale)
    return weibull


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def pareto(key, b, shape=None, dtype="float64"):
    seed = _get_seed(key)
    if shape is None:
        shape = b.shape
    # Draw samples from exponential distribution
    uniform = ivy.random_uniform(seed=seed, shape=shape, dtype=dtype)
    e = -ivy.log(1 - uniform)

    return ivy.exp(e / b)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
                "float16",
                "bfloat16",
        )
    },
    "jax",
)
def maxwell(key, shape=None, dtype="float64"):
    seed = _get_seed(key)
    # generate uniform random numbers between 0 and 1
    z = ivy.random_uniform(seed=seed, shape=shape, dtype=dtype)
    # applying inverse transform sampling
    x = (z ** 2) * ivy.exp(-(z ** 2) / 2)
    return x


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "0.4.13 and below": (
                "float32",
                "float64",
        )
    },
    "jax",
)
def ball(key, d, p=2.0, shape=(), dtype="float64"):
    seed = _get_seed(key)
    d = operator.index(d)

    g = ivy.gamma(1 / p, 1.0, shape=shape, dtype=dtype, seed=seed)
    b = ivy.bernoulli(ivy.array([0.5]), shape=shape, dtype=dtype, seed=seed)
    r = 2 * b - 1
    gn = r * g ** (1 / p)

    uniform = ivy.random_uniform(seed=seed, shape=shape, dtype=dtype)
    exp = -ivy.log(1 - uniform)

    return gn / (((ivy.abs(gn) ** p).sum(axis=-1) + exp) ** (1 / p))[..., None]


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "0.4.13 and below": (
                "float32",
                "float64",
        )
    },
    "jax",
)
def multivariate_normal(key, mean, cov, shape=(), dtype="float64", method="cholesky"):
    seed = _get_seed(key)

    if shape is None:
        shape = ivy.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])

    if method == 'eigh':
        (w, v) = ivy.eigh(cov)
        cov_factor = w * ivy.sqrt(w[..., None, :])
    elif method == 'svd':
        (u, s, _) = ivy.svd(cov)
        cov_factor = u * ivy.sqrt(s[..., None, :])
    else:
        cov_factor = ivy.cholesky(cov)

    rand_normal = ivy.random_normal(seed, shape=shape + mean.shape[-1:], dtype=dtype)
    result = mean + ivy.einsum('...ij,...j->...i', cov_factor, rand_normal)
    return result
