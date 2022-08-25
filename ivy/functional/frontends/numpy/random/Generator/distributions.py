import ivy

def uniform(a, b):
    return ivy.random_uniform(a, b)



def normal(loc=0.0, scale=1.0, size=None):
    ret = ivy.random_normal(mean=loc,
                             std=scale,
                             shape=size)
    return ret



def randint(low, high, size, dtype):
    return ivy.randint(low = low,high = high, shape = size, dtype= dtype)



def shuffle(x):
    return ivy.shuffle(x)



def seed(seed=None):
    ivy.seed(seed_value=seed)
