import ivy
from ivy.functional.frontends.mxnet.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def shuffle(x, axis=0):
    ivy.shuffle(x, axis)


@to_ivy_arrays_and_back
def normal(loc=0.0, scale=1.0, size=None, dtype=None, device=None, out=None):
    return ivy.random_normal(
        mean=loc, std=scale, shape=size, device=device, dtype=dtype, out=out
    )


@to_ivy_arrays_and_back
def randint(low, high=None, size=None, dtype=None, device=None, out=None):
    return ivy.randint(low, high=high, shape=size, device=device, dtype=dtype, out=out)


@to_ivy_arrays_and_back
def uniform(low=0.0, high=1.0, size=None, dtype=None, device=None, out=None):
    return ivy.random_uniform(
        low=low, high=high, shape=size, device=device, dtype=dtype, out=out
    )


@to_ivy_arrays_and_back
def rand(*size, **kwargs):
    return ivy.random_uniform(shape=size, **kwargs)


@to_ivy_arrays_and_back
def beta(a, b, size=None, dtype=None, device=None):
    return ivy.experimental.beta(a, b, shape=size, dtype=dtype, device=device)


@to_ivy_arrays_and_back
def chisquare(df, size=None, dtype=None, device=None):
    return ivy.experimental.gamma(
        df * 0.5,
        0.5,
        shape=size,
        dtype=dtype,
        device=device,
    )


@to_ivy_arrays_and_back
def gamma(shape, scale=1.0, size=None, dtype=None, device=None, out=None):
    return ivy.experimental.gamma(
        shape, scale, shape=size, dtype=dtype, device=device, out=out
    )


@to_ivy_arrays_and_back
def multinomial(n, pvals, size=None, **kwargs):
    num_samples = ivy.prod(size)
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
    return ivy.multinomial(n, num_samples, batch_size=batch_size, probs=pvals, **kwargs)


@to_ivy_arrays_and_back
def power(a, size=None, dtype=None, device=None, out=None):
    # special case of beta function
    b = ivy.ones_like(a)
    return ivy.experimental.beta(a, b, shape=size, dtype=dtype, device=device, out=out)
