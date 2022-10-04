# local
import ivy


def random(size=None):
    return ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")


def multinomial(n, p_vals, size=None):
    num_samples = len(p_vals)
    return ivy.multinomial(n, num_samples, batch_size=size, probs=ivy.array(p_vals))
