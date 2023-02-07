# local
import ivy.functional.frontends.numpy as ivy_np


def default__rng(seed=None):
    return Generator(seed=seed)


class Generator:
    def __init__(self, bit_generator=None):
        self.seed = bit_generator

    def multinomial(self, n, pvals, size=None):
        ivy_np.random.multinomial(n, pvals, size=size)
