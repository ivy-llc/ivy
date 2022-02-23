#global
import jax.numpy as jx

def det(x:jx.array) -> jx.array:
    return jx.linalg.det(x)