"""
Collection of tests for the demos
"""

# global
import pytest

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# Tests #
# ------#

# training
def test_training_demo(dev_str, call):

    if call is helpers.np_call:
        # numpy does not support gradients
        pytest.skip()

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear2 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear2(x))

    model = MyModel()
    optimizer = ivy.Adam(1e-4)
    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.reduce_mean((out - target) ** 2)[0]

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)


# functional api
def test_array(dev_str, call):
    ivy.unset_framework()
    import jax.numpy as jnp
    assert ivy.concatenate((jnp.ones((1,)), jnp.ones((1,))), -1).shape == (2,)
    import tensorflow as tf
    assert ivy.concatenate((tf.ones((1,)), tf.ones((1,))), -1).shape == (2,)
    import numpy as np
    assert ivy.concatenate((np.ones((1,)), np.ones((1,))), -1).shape == (2,)
    import mxnet as mx
    assert ivy.concatenate((mx.nd.ones((1,)), mx.nd.ones((1,))), -1).shape == (2,)
    import torch
    assert ivy.concatenate((torch.ones((1,)), torch.ones((1,))), -1).shape == (2,)
