import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back

try:
    from torch import Generator
except ImportError:
    from types import SimpleNamespace

    Generator = SimpleNamespace


def seed() -> int:
    """Returns a 64 bit number used to seed the RNG"""
    return int(ivy.randint(-(2**63), 2**63 - 1))


@to_ivy_arrays_and_back
def manual_seed(seed: int):
    ivy.seed(seed_value=seed)
    return Generator().manual_seed(seed)


@with_supported_dtypes(
    {
        "1.11.0 and below": (
            "float32",
            "float64",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    return ivy.multinomial(
        num_samples + 1,  # doesn't matter because `probs` is provided, but should be
        # greater than the number of samples
        num_samples,
        probs=input,
        replace=replacement,
        out=out,
    )


@with_supported_dtypes(
    {
        "1.11.0 and below": (
            "float32",
            "float64",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def poisson(input, generator=None):
    return ivy.poisson(input, shape=None)


@to_ivy_arrays_and_back
def rand(
    size,
    *,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False
):
    return ivy.random_uniform(
        shape=size,
        out=out,
        dtype=dtype,
        device=device,
    )
