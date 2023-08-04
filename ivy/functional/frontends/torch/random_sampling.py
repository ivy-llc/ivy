import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_gradients,
)


# ToDo: will need to create a Generator class to be able to fully test these functions


def seed() -> int:
    """Return a 64 bit number used to seed the RNG."""
    return int(ivy.randint(-(2**63), 2**63 - 1))


@to_ivy_arrays_and_back
def manual_seed(seed: int):
    ivy.seed(seed_value=seed)
    return None


@with_supported_dtypes(
    {
        "2.0.1 and below": (
            "float32",
            "float64",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    seed = generator.initial_seed() if generator is not None else None
    return ivy.multinomial(
        num_samples + 1,  # doesn't matter because `probs` is provided, but should be
        # greater than the number of samples
        num_samples,
        probs=input,
        replace=replacement,
        seed=seed,
        out=out,
    )


@with_supported_dtypes(
    {
        "2.0.1 and below": (
            "float32",
            "float64",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def poisson(input, generator=None):
    seed = generator.initial_seed() if generator is not None else None
    return ivy.poisson(input, seed=seed, shape=None)


@to_ivy_arrays_and_back
def randint(
    low,
    high,
    size,
    *,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    seed = generator.initial_seed() if generator is not None else None
    return ivy.randint(
        low,
        high,
        shape=size,
        seed=seed,
        out=out,
        dtype=dtype,
        device=device,
    )


@to_ivy_arrays_and_back
def rand(
    *size,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
    **kwargs,
):
    if not size and "size" not in kwargs:
        raise ValueError("Missing 1 required positional/keyword argument: size")
    size = kwargs["size"] if not size else size
    if (
        isinstance(size, (list, tuple))
        and len(size) == 1
        and isinstance(size[0], (list, tuple))
    ):
        size = size[0]
    seed = generator.initial_seed() if generator is not None else None
    return ivy.random_uniform(
        shape=size,
        seed=seed,
        out=out,
        dtype=dtype,
        device=device,
    )


@with_supported_dtypes(
    {
        "2.0.1 and below": (
            "float32",
            "float64",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def normal(mean, std, *, generator=None, out=None):
    seed = generator.initial_seed() if generator is not None else None
    return ivy.random_normal(mean=mean, std=std, seed=seed, out=out)


@to_ivy_arrays_and_back
def rand_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=False,
):
    shape = input.shape
    if not dtype:
        dtype = input.dtype

    return ivy.random_uniform(
        shape=shape,
        dtype=dtype,
        device=device,
    )


@handle_gradients
@to_ivy_arrays_and_back
def randn(
    *size,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
    **kwargs,
):
    if not size and "size" not in kwargs:
        raise ValueError("Missing 1 required positional/keyword argument: size")
    size = kwargs["size"] if not size else size
    if (
        isinstance(size, (list, tuple))
        and len(size) == 1
        and isinstance(size[0], (list, tuple))
    ):
        size = size[0]
    seed = generator.initial_seed() if generator is not None else None
    return ivy.random_normal(
        shape=size,
        seed=seed,
        out=out,
        dtype=dtype,
        device=device,
    )


@to_ivy_arrays_and_back
def randn_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    shape = input.shape
    if not dtype:
        dtype = input.dtype

    return ivy.random_normal(
        shape=shape,
        dtype=dtype,
        device=device,
    )


@with_supported_dtypes(
    {
        "2.0.1 and below": (
            "float32",
            "float64",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def bernoulli(input, *, generator=None, out=None):
    seed = generator.initial_seed() if generator is not None else None
    return ivy.bernoulli(input, seed=seed, out=out)


@to_ivy_arrays_and_back
def randperm(
    n,
    *,
    generator=None,
    out=None,
    dtype=ivy.int64,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    seed = generator.initial_seed() if generator is not None else None
    arr = ivy.arange(n, device=device, dtype=dtype)
    ret = ivy.shuffle(arr, seed=seed, out=out)
    return ret


@to_ivy_arrays_and_back
def randint_like(
    input,
    low,
    high,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    shape = input.shape

    return ivy.randint(
        low,
        high,
        shape=shape,
        device=device,
        dtype=dtype,
    )
