#global import
import ivy
import jax
from jax.config import config
from typing import Optional, Union, Sequence


def __init__(self):
    self.key = PRNGKey(0)
def _setRNG(key):
    global RNG
    RNG.key = key



def PRNGKey() ->ivy.Array:
    ivy.seed(seed_value=RNG.key)
    return ivy.randint(RNG.key)
    
