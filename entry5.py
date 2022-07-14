import ivy
import tensorflow as tf
import jax.numpy as jnp
import numpy as np


x = np.random.normal(size=(3,2,2))
y = np.random.normal(size=(2,2))

# print("second vmap testing")
# ivy.set_backend("numpy")
# print("numpy vmap:", ivy.vmap(ivy.matmul, (None, None))(ivy.array([[0, 0], [0, 0]]), ivy.array([[0, 0], [0, 0]])))
# ivy.unset_backend()
ivy.set_backend("jax")
print("jax vmap:", ivy.vmap(ivy.matmul, (None, None))(ivy.array([[0, 0], [0, 0]]), ivy.array([[0, 0], [0, 0]])))