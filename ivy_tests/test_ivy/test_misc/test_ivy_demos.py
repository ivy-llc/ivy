"""Collection of tests for the demos."""

# global
import pytest

# local
import ivy
import ivy.functional.backends.numpy


# Tests #
# ------#

# training
def test_training_demo(on_device):

    if ivy.current_backend_str() == "numpy":
        # numpy does not support gradients
        pytest.skip()

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))

    model = MyModel()
    optimizer = ivy.Adam(1e-4)
    x_in = ivy.array([1.0, 2.0, 3.0])
    target = ivy.array([0.0])

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.mean((out - target) ** 2)

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)


# functional api
def test_array(on_device):
    ivy.unset_backend()
    import jax.numpy as jnp

    assert ivy.concat((jnp.ones((1,)), jnp.ones((1,))), axis=-1).shape == (2,)
    import tensorflow as tf

    assert ivy.concat((tf.ones((1,)), tf.ones((1,))), axis=-1).shape == (2,)
    import numpy as np

    assert ivy.concat((np.ones((1,)), np.ones((1,))), axis=-1).shape == (2,)
    import torch

    assert ivy.concat((torch.ones((1,)), torch.ones((1,))), axis=-1).shape == (2,)
