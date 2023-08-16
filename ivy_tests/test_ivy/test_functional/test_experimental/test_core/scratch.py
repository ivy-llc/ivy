import ivy
import numpy as np
import tensorflow as tf
import torch
import jax.numpy as jnp
import paddle

ivy.set_backend("jax")
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

# def sparsify_tensor(tensor, card):
#     if card >= np.prod(tensor.shape):
#         return tensor
#     bound = np.reshape(np.abs(tensor), (-1, ))[-card]

#     return np.where(
#         np.abs(tensor) < bound,
#         np.zeros_like(tensor),
#         tensor,
#     )


# def sparsify_tensor(tensor, card):
#     if card >= tf.math.reduce_prod(tensor.shape):
#         return tensor
#     bound = tf.reshape(tf.abs(tensor), (-1, ))[-card]

#     return tf.where(
#         tf.abs(tensor) < bound,
#         tf.zeros_like(tensor),
#         tensor,
#     )


# def sparsify_tensor(tensor, card):
#     if card >= torch.prod(tensor.shape):
#         return tensor
#     bound = torch.sort(torch.abs(tensor), axis=None)[-card]

#     return torch.where(
#         torch.abs(tensor) < bound,
#         torch.zeros_like(tensor),
#         tensor,
#     )

# print(sparsify_tensor(x, 100))
