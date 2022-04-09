
import numpy as np

def modify_and_create_array(x):
    x += 5
    x *= 2
    x -= 5
    x /= 10
    return np.asarray(x)