import ivy
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


@to_ivy_arrays_and_back
def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    return ivy.multinomial(input, num_samples, replace=replacement, out=out)
