# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
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
        "0.3.14 and below": (
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
        "0.3.14 and below": (
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
    {"0.3.14 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def poisson(key, lam, shape=None, dtype=None):
    seed = _get_seed(key)
    return ivy.poisson(lam, shape=shape, dtype=dtype, seed=seed)


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
def gamma(key, a, shape=None, dtype="float64"):
    seed = _get_seed(key)
    return ivy.gamma(a, 1.0, shape=shape, dtype=dtype, seed=seed)


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
    {"0.3.14 and below": ("unsigned", "int8", "int16")},
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
        "0.3.14 and below": (
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
    {"0.3.14 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def randint(key, shape, minval, maxval, dtype="int64"):
    seed = _get_seed(key)
    return ivy.randint(minval, maxval, shape=shape, dtype=dtype, seed=seed)


@handle_jax_dtype
@to_ivy_arrays_and_back
def multivariate_normal(key, mean, cov, shape=None, dtype=None, method="cholesky"):
    _get_seed(key)
    if method not in {"svd", "eigh", "cholesky"}:
        raise ValueError("method must be one of {'svd', 'eigh', 'cholesky'}")
    if not ivy.get_num_dims(mean) >= 1:
        msg = "multivariate_normal requires mean.ndim >= 1, got mean.ndim == {}"
        raise ValueError(msg.format(ivy.get_num_dims(mean)))
    if not ivy.get_num_dims(cov) >= 2:
        msg = "multivariate_normal requires cov.ndim >= 2, got cov.ndim == {}"
        raise ValueError(msg.format(ivy.get_num_dims(cov)))

    n = mean.shape[-1]

    if ivy.shape(cov)[-2:] != (n, n):
        msg = (
            "multivariate_normal requires cov.shape == (..., n, n) for n={n}, "
            "but got cov.shape == {shape}."
        )
        raise ValueError(msg.format(n=n, shape=ivy.shape(cov)))

    if shape is None:
        shape = ivy.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])

    if method == "svd":
        (u, s, _) = ivy.svd(cov)
        factor = u * ivy.sqrt(s[..., None, :])
    elif method == "eigh":
        (w, v) = ivy.eigh(cov)
        factor = v * ivy.sqrt(w[..., None, :])
    else:  # 'cholesky'
        factor = ivy.cholesky(cov)
    normal_samples = normal(key, shape + mean.shape[-1:], dtype)
    result = mean + ivy.einsum("...ij,...j->...i", factor, normal_samples)
    return result
