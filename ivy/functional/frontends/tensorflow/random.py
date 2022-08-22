# global

import ivy

def shuffle(value, seed = None, name = None):
    ivy.seed(seed_value = seed)
    return ivy.shuffle(value, out = name)
