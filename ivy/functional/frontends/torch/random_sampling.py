import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


def seed() -> int:
    """Returns a 64 bit number used to seed the RNG"""
    return int(ivy.randint(-(2**63), 2**63 - 1))


def manual_seed(seed: int) -> None:
    return ivy.seed(seed_value=seed)


@to_ivy_arrays_and_back
def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    return ivy.multinomial(input, num_samples, replace=replacement, out=out)
