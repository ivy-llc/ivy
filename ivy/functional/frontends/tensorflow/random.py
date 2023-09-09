import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
)
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
def gamma(shape, alpha, beta=None, dtype=ivy.float32, seed=None, name=None):
    return ivy.gamma(alpha, beta, shape=shape, dtype=dtype, seed=seed)


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int8", "int16", "int32", "int64", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def normal(shape, mean=0.0, stddev=1.0, dtype=ivy.float32, seed=None, name=None):
    return ivy.random_normal(mean=mean, std=stddev, shape=shape, dtype=dtype, seed=seed)


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
@handle_tf_dtype
def poisson(shape, lam, dtype=ivy.float32, seed=None, name=None):
    shape = ivy.array(shape, dtype=ivy.int32)
    lam = ivy.array(lam, dtype=ivy.float32)
    if lam.ndim > 0:
        shape = ivy.concat([shape, ivy.array(lam.shape)])
    return ivy.poisson(shape=shape, lam=lam, dtype=dtype, seed=seed, fill_value=0)


# implement random shuffle
@with_unsupported_dtypes(
    {"2.13.0 and below": ("int8", "int16", "in32", "int64", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def shuffle(value, seed=None, name=None):
    return ivy.shuffle(value, seed=seed)


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def stateless_binomial(counts, probs, seed, output_dtype=ivy.int32, shape=None, name=None):
    if hasattr(probs, '__iter__'):
        if any(p < 0 or p > 1 for p in probs):
            raise ValueError("probs elements must be in the interval (0, 1)")
    else:
        if probs < 0 or probs > 1:
            raise ValueError("probs must be in the interval (0, 1)")
    if hasattr(counts, '__iter__'):
        if any(c < 0 for c in counts):
            raise ValueError("counts elemenst must be strictly positive")
    else:
        if counts < 0:
            raise ValueError("counts must be strictly positive")
    if isinstance(shape, int):
        shape = (shape,)
    lambda_ = ivy.multiply(counts, probs)
    return ivy.poisson(lam=lambda_, shape=shape, seed=seed[0] + seed[1], dtype=output_dtype)


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def stateless_gamma(shape, seed, alpha, beta=None, dtype=ivy.float32, name=None):
    return ivy.gamma(shape=shape, seed=seed[0] + seed[1], alpha=alpha, beta=beta, dtype=dtype)


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def stateless_normal(
    shape, seed, mean=0.0, stddev=1.0, dtype=ivy.float32, name=None, alg="auto_select"
):
    return ivy.random_normal(
        mean=mean, std=stddev, shape=shape, dtype=dtype, seed=seed[0] + seed[1]
    )


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def stateless_poisson(shape, seed, lam, dtype=ivy.int32, name=None):
    return ivy.poisson(shape=shape, lam=lam, dtype=dtype, seed=seed[0] + seed[1])


@to_ivy_arrays_and_back
def stateless_uniform(
    shape, seed, minval=0, maxval=None, dtype=ivy.float32, name=None, alg="auto_select"
):
    return ivy.random_uniform(
        shape=shape, seed=seed[0] + seed[1], low=minval, high=maxval, dtype=dtype
    )


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_ivy_arrays_and_back
def uniform(shape, minval=0, maxval=None, dtype=ivy.float32, seed=None, name=None):
    if maxval is None:
        if dtype != "int64":
            maxval = 1.0
        else:
            raise ValueError("maxval must be specified for int64 dtype")
    return ivy.random_uniform(
        shape=shape, low=minval, high=maxval, dtype=dtype, seed=seed
    )
