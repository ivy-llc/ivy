# global
import operator

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_jax_dtype,
)


# --- Helpers --- #
# --------------- #


def _get_seed(key):
    if "PRNGKeyArray" in repr(key):
        key = key._base_array
    key1, key2 = int(key[0]), int(key[1])
    return ivy.to_scalar(int("".join(map(str, [key1, key2]))))


def _remove_axis(shape, axis):
    return shape[:axis] + shape[axis + 1 :]


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
def PRNGKey(seed, *, impl=None):
    return ivy.array([0, seed % 4294967295 - (seed // 4294967295)], dtype=ivy.int64)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "0.4.24 and below": (
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


@to_ivy_arrays_and_back
def bernoulli(key, p=0.5, shape=None):
    seed = _get_seed(key)
    return ivy.bernoulli(p, shape=shape, seed=seed)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def beta(key, a, b, shape=None, dtype=None):
    seed = _get_seed(key)
    return ivy.beta(a, b, shape=shape, dtype=dtype, seed=seed)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def categorical(key, logits, axis, shape=None):
    logits_arr = ivy.asarray(logits)

    if axis >= 0:
        axis -= len(logits_arr.shape)
    batch_shape = tuple(_remove_axis(logits_arr.shape, axis))

    if shape is None:
        shape = batch_shape
    else:
        shape = tuple(shape)
        if shape != batch_shape:
            raise ValueError(
                +f"Shape {shape} is not compatible with reference shape {batch_shape}"
            )

    logits_shape = list(shape[len(shape) - len(batch_shape) :])
    logits_shape.insert(axis % len(logits_arr.shape), logits_arr.shape[axis])

    gumbel_noise = gumbel(key, ivy.array(logits_shape), logits_arr.dtype)
    expanded_logits = ivy.expand_dims(logits_arr, axis=axis)
    noisy_logits = gumbel_noise + expanded_logits

    # Use Ivy's argmax to get indices
    indices = ivy.argmax(noisy_logits, axis=axis)

    return indices


@handle_jax_dtype
@to_ivy_arrays_and_back
def cauchy(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    u = ivy.random_uniform(low=0.0, high=1.0, shape=shape, dtype=dtype, seed=seed)
    return ivy.tan(ivy.pi * (u - 0.5))


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
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
@with_unsupported_dtypes(
    {"0.4.24 and below": "uint32"},
    "jax",
)
def double_sided_maxwell(key, loc, scale, shape=(), dtype="float64"):
    params_shapes = ivy.broadcast_shapes(ivy.shape(loc), ivy.shape(scale))
    if not shape:
        shape = params_shapes

    shape = shape + params_shapes
    maxwell_rvs = maxwell(key, shape=shape, dtype=dtype)
    random_sign = rademacher(key, shape=shape, dtype=dtype)

    return random_sign * maxwell_rvs * scale + loc


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
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


@to_ivy_arrays_and_back
def fold_in(key, data):
    if "PRNGKeyArray" in repr(key):
        key = key._base_array
    s = ivy.bitwise_left_shift(
        ivy.asarray(data, dtype=ivy.uint32), ivy.array(32, dtype=ivy.uint32)
    )
    return ivy.bitwise_xor(key, s)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
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
        "0.4.24 and below": (
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


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
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


# loggamma
@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def loggamma(key, a, shape=None, dtype="float64"):
    seed = _get_seed(key)
    return ivy.log(ivy.gamma(a, 1.0, shape=shape, dtype=dtype, seed=seed))


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.24 and below": ("float16", "bfloat16")},
    "jax",
)
def logistic(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform_x = ivy.random_uniform(seed=seed, shape=shape, dtype=dtype)
    return ivy.log(ivy.divide(uniform_x, ivy.subtract(1.0, uniform_x)))


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
def maxwell(key, shape, dtype="float64"):
    seed = _get_seed(key)
    shape = shape + (3,)
    random_normal = ivy.random_normal(seed=seed, shape=shape, dtype=dtype)
    return ivy.vector_norm(random_normal, axis=-1)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def multivariate_normal(key, mean, cov, shape=None, dtype="float64", method="cholesky"):
    if shape is None:
        shape = ivy.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
    if method == "cholesky":
        cov_factor = ivy.cholesky(cov)
    elif method == "eigh":
        (w, v) = ivy.eigh(cov)
        cov_factor = v * ivy.sqrt(w[..., None, :])
    elif method == "svd":
        (u, s, _) = ivy.svd(cov)
        cov_factor = u * ivy.sqrt(s[..., None, :])

    rand_normal = normal(key=key, shape=shape + mean.shape[-1:], dtype=dtype)
    result = mean + ivy.einsum("...ij,...j->...i", cov_factor, rand_normal.ivy_array)

    return result


@handle_jax_dtype
@to_ivy_arrays_and_back
def normal(key, shape=(), dtype=None):
    seed = _get_seed(key)
    return ivy.random_normal(shape=shape, dtype=dtype, seed=seed)


@handle_jax_dtype
@to_ivy_arrays_and_back
def orthogonal(key, n, shape=(), dtype=None):
    seed = _get_seed(key)
    flat_shape = (n, n)
    if shape:
        flat_shape = shape + flat_shape

    # Generate a random matrix with the given shape and dtype
    random_matrix = ivy.random_uniform(seed=seed, shape=flat_shape, dtype=dtype)

    # Compute the QR decomposition of the random matrix
    q, _ = ivy.linalg.qr(random_matrix)

    # Reshape the resulting orthogonal matrix to the desired shape
    if shape:
        q = ivy.reshape(q, shape + (n, n))

    return q


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
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


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.24 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def poisson(key, lam, shape=None, dtype=None):
    seed = _get_seed(key)
    return ivy.poisson(lam, shape=shape, dtype=dtype, seed=seed, fill_value=-1)


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.24 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def rademacher(key, shape, dtype="int64"):
    seed = _get_seed(key)
    prob = ivy.full(shape, 0.5, dtype="float32")
    b = ivy.bernoulli(prob, shape=shape, dtype="float32", seed=seed)
    b = ivy.astype(b, dtype)
    return 2 * b - 1


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.24 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def randint(key, shape, minval, maxval, dtype="int64"):
    seed = _get_seed(key)
    return ivy.randint(minval, maxval, shape=shape, dtype=dtype, seed=seed)


@to_ivy_arrays_and_back
def shuffle(key, x, axis=0):
    seed = _get_seed(key)
    x = ivy.flip(x, axis=axis)
    return ivy.shuffle(x, seed=seed)


@handle_jax_dtype
@to_ivy_arrays_and_back
def t(key, df, shape=(), dtype="float64"):
    seed = _get_seed(key)
    n = ivy.random_normal(shape=shape, dtype=dtype, seed=seed)
    half_df = df / 2.0
    g = ivy.gamma(half_df, 1.0, shape=shape, dtype=dtype, seed=seed)
    return n * ivy.sqrt(ivy.divide(half_df, g))


@handle_jax_dtype
@to_ivy_arrays_and_back
def uniform(key, shape=(), dtype=None, minval=0.0, maxval=1.0):
    seed = _get_seed(key)
    return ivy.random_uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, seed=seed
    )


@handle_jax_dtype
@to_ivy_arrays_and_back
def weibull_min(key, scale, concentration, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform_x = ivy.random_uniform(seed=seed, shape=shape, dtype=dtype)
    x = 1 - uniform_x
    weibull = x ** (concentration - 1) * -ivy.log(x / scale)
    return weibull
