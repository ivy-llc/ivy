import ivy
import ivy.numpy as ivy_np
import jax
import jax.numpy as jnp
import tensorflow as tf
import torch
import paddle
from ivy_tests.ndarrays_tests import ndarray_test_util as tu


def test_frontend_ndarray_property_flat():

    # Create a random Ivy array
    ivy_arr = ivy.random_uniform((2, 3, 4), -1.0, 1.0)

    # Convert Ivy array to NumPy array
    numpy_arr = ivy_np.to_numpy(ivy_arr)

    # Convert Ivy array to JAX array
    jax_arr = jax.numpy.asarray(ivy.to_numpy(ivy_arr))

    # Convert Ivy array to TensorFlow tensor
    tf_tensor = tf.convert_to_tensor(ivy.to_numpy(ivy_arr))

    # Convert Ivy array to PyTorch tensor
    torch_tensor = torch.tensor(ivy.to_numpy(ivy_arr))

    # Convert Ivy array to PaddlePaddle tensor
    paddle_tensor = paddle.to_tensor(ivy.to_numpy(ivy_arr))

    # Flatten Ivy and NumPy arrays
    ivy_flat = ivy_np.ndarray_property_flat(ivy_arr)
    numpy_flat = numpy_arr.flatten()
    jax_flat = jnp.ndarray.flatten(jax_arr)
    tf_flat = tf.reshape(tf_tensor, [-1])
    torch_flat = torch_tensor.flatten()
    paddle_flat = paddle_tensor.flatten()

    # Convert Ivy flat array to NumPy array
    ivy_flat_np = ivy_np.to_numpy(ivy_flat)

    # Compare the flat arrays
    tu.assert_array_equal(ivy_flat_np.flatten(), numpy_flat)
    tu.assert_array_equal(ivy_flat_np, jax_flat)
    tu.assert_array_equal(ivy_flat_np, tf_flat.numpy())
    tu.assert_array_equal(ivy_flat_np, torch_flat.numpy())
    tu.assert_array_equal(ivy_flat_np, paddle_flat.numpy())

