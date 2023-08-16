import ivy
import numpy as np
import tensorflow as tf
import torch
import jax.numpy as jnp
import paddle

x = ivy.arange(1000)
x = ivy.reshape(x, (100, 10))
x = ivy.sparsify_tensor(x, 100)
print(x)

ivy.set_backend("torch")
x = torch.arange(1000)
x = torch.reshape(x, (100, 10))
x = ivy.sparsify_tensor(x, 100)
print(x)

ivy.set_backend("jax")
x = jnp.arange(1000)
x = jnp.reshape(x, (100, 10))
x = ivy.sparsify_tensor(x, 100)
print(x)

ivy.set_backend("numpy")
x = np.arange(1000)
x = np.reshape(x, (100, 10))
x = ivy.sparsify_tensor(x, 100)
print(x)

ivy.set_backend("tensorflow")
x = tf.range(1000)
x = tf.reshape(x, (100, 10))
x = ivy.sparsify_tensor(x, 100)
print(x)

ivy.set_backend("paddle")
x = paddle.arange(1000)
x = paddle.reshape(x, (100, 10))
x = ivy.sparsify_tensor(x, 100)
print(x)
