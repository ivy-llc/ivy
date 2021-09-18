"""
Collection of tests for training neural network layers with "on call" building
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


# Weight Conditioned Network #
# ---------------------------#

class FC(ivy.Module):

    def __init__(self, output_size=1, num_layers=2, layer_dim=64, dev_str='cpu', v=None):
        self._output_size = output_size
        self._num_layers = num_layers
        self._layer_dim = layer_dim
        self._dev_str = dev_str
        super(FC, self).__init__(v=v,  build_mode='on_call')

    # noinspection PyUnusedLocal
    def _build(self, x, *args, **kwargs):
        input_size = x.shape[-1]
        self._layers = [ivy.Linear(input_size, self._layer_dim, dev_str=self._dev_str)]
        self._layers += [ivy.Linear(self._layer_dim, self._layer_dim, dev_str=self._dev_str)
                         for _ in range(self._num_layers - 2)]
        self._layers.append(ivy.Linear(self._layer_dim, self._output_size, dev_str=self._dev_str))

    def _forward(self, x):
        for layer in self._layers:
            x = ivy.leaky_relu(layer(x))
        return x


class WeConLayerFC(ivy.Module):

    def __init__(self, num_layers=2, layer_dim=64, v=None):
        self._num_layers = num_layers
        self._layer_dim = layer_dim
        super(WeConLayerFC, self).__init__(v=v,  build_mode='on_call')

    # noinspection PyUnusedLocal
    def _build(self, implicit_weights, *args, **kwargs):
        implicit_shapes = implicit_weights.shapes
        self._layers = list()
        for i in range(self._num_layers):
            if i == 0:
                self._layers.append(implicit_shapes.map(
                    lambda shp, kc: ivy.Linear(int(np.prod(shp[1:])), self._layer_dim, dev_str=self._dev_str)))
            else:
                self._layers.append(implicit_shapes.map(
                    lambda shp, kc: ivy.Linear(self._layer_dim, self._layer_dim, dev_str=self._dev_str)))

    def _forward(self, implicit_weights):
        xs = implicit_weights
        for layer in self._layers:
            xs = ivy.Container.multi_map(lambda args, _: ivy.leaky_relu(args[0](args[1])), [layer, xs])
        return xs


class WeConFC(ivy.Module):

    def __init__(self, dev_str='cpu', v=None):
        self._layer_specific_fc = WeConLayerFC()
        self._fc = FC()
        self._dev_str = dev_str
        super(WeConFC, self).__init__(v=v, build_mode='on_call')

    # noinspection PyUnusedLocal
    def _build(self, *args, **kwargs):
        # both child layers are built on_call
        pass

    def _forward(self, implicit_weights):
        batch_shape = [i for i in implicit_weights.shape if i]
        total_batch_size = np.prod(batch_shape)
        reshaped_weights = implicit_weights.reshape(pre_shape=[total_batch_size], post_shape=[-1])
        xs = self._layer_specific_fc(reshaped_weights)
        x = ivy.concatenate([v for k, v in xs.to_iterator()], -1)
        ret_flat = self._fc(x)
        return ivy.reshape(ret_flat, batch_shape + [-1])


# WeConFC
@pytest.mark.parametrize(
    "batch_shape", [[1, 2]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_weight_conditioned_network_training(batch_shape, dtype_str, tensor_fn, dev_str, call):

    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    x = ivy.Container({'layer0': {'w': ivy.random_uniform(shape=batch_shape+[64, 3]),
                                  'b': ivy.random_uniform(shape=batch_shape+[64])},
                       'layer1': {'w': ivy.random_uniform(shape=batch_shape+[1, 64]),
                                  'b': ivy.random_uniform(shape=batch_shape+[1])}})
    we_con_net = WeConFC()

    def loss_fn(v_=None):
        out = we_con_net(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    loss_fn()  # build on_call layers
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, we_con_net.v)
        we_con_net.v = ivy.gradient_descent_update(we_con_net.v, grads, 1e-3)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    helpers.assert_compilable(loss_fn)
