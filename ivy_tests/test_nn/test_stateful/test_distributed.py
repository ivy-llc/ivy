"""
Collection of tests for Ivy distributed training
"""

# global
import copy
import pytest

# local
import ivy
import ivy_tests.helpers as helpers


class TrainableModule(ivy.Module):

    def __init__(self, in_size, out_size, dev_str=None, build_mode='explicit', hidden_size=64, store_vars=True):
        self._in_size = in_size
        self._out_size = out_size
        self._hidden_size = hidden_size
        ivy.Module.__init__(self, dev_str, build_mode=build_mode, store_vars=store_vars)

    def _build(self):
        self._linear0 = ivy.Linear(self._in_size, self._hidden_size, dev_str=self._dev_str)
        self._linear1 = ivy.Linear(self._hidden_size, self._hidden_size, dev_str=self._dev_str)
        self._linear2 = ivy.Linear(self._hidden_size, self._out_size, dev_str=self._dev_str)

    def _forward(self, x):
        x = ivy.expand_dims(x, 0)
        x = ivy.tanh(self._linear0(x))
        x = ivy.tanh(self._linear1(x))
        return ivy.tanh(self._linear2(x))[0]


# distributed training
@pytest.mark.parametrize(
    "bs_ic_oc", [([2, 1], 4, 5)])
def test_distributed_training(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()

    # devices
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    dev_strs = [dev_str0, dev_str1]

    # input
    batch_shape, input_channels, output_channels = bs_ic_oc
    dev_batch_shape = [int(batch_shape[0]/2)] + batch_shape[1:]
    x0 = ivy.cast(ivy.linspace(ivy.zeros(dev_batch_shape), ivy.ones(dev_batch_shape),
                               input_channels, dev_str=dev_str0), 'float32')
    x1 = ivy.cast(ivy.linspace(ivy.zeros(dev_batch_shape), ivy.ones(dev_batch_shape),
                               input_channels, dev_str=dev_str1), 'float32')
    x = ivy.Distributed([x0, x1])

    # module
    module = TrainableModule(input_channels, output_channels, dev_str=dev_str0)
    module.build()

    # optimizer
    optim = ivy.SGD(1e-4)

    # loss
    def loss_fn(x_, v_):
        out = module(x_, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss_n_grads = ivy.MultiDevice(
            ivy.map(lambda xn, vc: ivy.execute_with_gradients(
                lambda v: loss_fn(xn, v), vc), x, module.v.clone(dev_strs)))
        loss, grads = ivy.unify_iter(loss_n_grads, dev_str0, 'mean')
        module.v = optim.step(module.v, grads)
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
    assert ivy.reduce_max(ivy.abs(grads.linear0.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear0.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear1.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear1.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear2.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear2.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


def loss_fn(module, x_, v_):
    out = module(x_, v=v_)
    return ivy.reduce_mean(out)[0]


def map_fn(module, xn, vc):
    return ivy.execute_with_gradients(lambda v: loss_fn(module, xn, v), vc)


# distributed multiprocess training
@pytest.mark.parametrize(
    "bs_ic_oc", [([2, 1], 4, 5)])
def test_distributed_multiprocess_training(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()

    if call is not helpers.torch_call:
        # ToDo: add support for other frameworks, currently only supported for torch
        pytest.skip()

    # devices
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    dev_strs = [dev_str0, dev_str1]

    # input
    batch_shape, input_channels, output_channels = bs_ic_oc
    dev_batch_shape = [int(batch_shape[0]/2)] + batch_shape[1:]
    x0 = ivy.cast(ivy.linspace(ivy.zeros(dev_batch_shape), ivy.ones(dev_batch_shape),
                               input_channels, dev_str=dev_str0), 'float32')
    x1 = ivy.cast(ivy.linspace(ivy.zeros(dev_batch_shape), ivy.ones(dev_batch_shape),
                               input_channels, dev_str=dev_str1), 'float32')
    x = ivy.Distributed([x0, x1])

    # module for processes
    module = TrainableModule(input_channels, output_channels, dev_str=dev_str0, store_vars=False)

    # optimizer
    optim = ivy.SGD(1e-4)

    # device manager
    dev_mapper = ivy.DevMapperMultiProc(map_fn, dev_strs, [copy.deepcopy(module) for _ in range(len(dev_strs))])

    # local module
    module = TrainableModule(input_channels, output_channels, dev_str=dev_str0, store_vars=True)
    module.build()

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss_n_grads = dev_mapper.map(x, module.v.clone(dev_strs))
        loss, grads = ivy.unify_iter(loss_n_grads, dev_str0, 'mean')
        module.v = optim.step(module.v, grads)
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
    assert ivy.reduce_max(ivy.abs(grads.linear0.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear0.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear1.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear1.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear2.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear2.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)
