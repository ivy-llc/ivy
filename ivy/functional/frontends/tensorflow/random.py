# global
import ivy

def shuffle(value, seed: int=None, name=None):

    ivy.seed(seed_value = seed)
    return ivy.shuffle(value, out=name)


def normal(mean=0.0,
           stddev=1.0,
           shape=None,
           dtype=None,
           seed=None,
           name='None'):

    if seed is not None:
        ivy.seed(seed_value = seed)

    return ivy.random_normal(mean=mean,
                             std=stddev,
                             shape=shape,
                             dtype=dtype,
                             device=None,
                             out=None)

