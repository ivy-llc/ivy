# local
import ivy


def random(size=None):
    return ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")


def multinomial(n, pvals, size=None):
    num_samples = len(pvals)
    return ivy.multinomial(n, num_samples, batch_size=size, probs=ivy.array(pvals))
