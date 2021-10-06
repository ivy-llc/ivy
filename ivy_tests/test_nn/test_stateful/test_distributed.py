"""
Collection of tests for Ivy distributed training
"""

# global
import pytest

# local
import ivy
import ivy_tests.helpers as helpers
from ivy_tests.test_nn.test_stateful.test_converters import NATIVE_MODULES


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


def loss_fn(module, x_, v_):
    out = module(x_, v=v_)
    return ivy.reduce_mean(out)[0]


def map_fn(module, dev_str, xn, vc):
    return ivy.execute_with_gradients(lambda v: loss_fn(module, xn, v), vc)


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
    x = ivy.DistributedItem([x0, x1])

    # module
    module = TrainableModule(input_channels, output_channels, dev_str=dev_str0)
    module.build()

    # optimizer
    optim = ivy.SGD(1e-4)

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss_n_grads = ivy.MultiDevIter(
            ivy.map(map_fn,
                    constant={'module': module, 'dev_str': dev_str0},
                    unique={'xn': x.at_devs(), 'vc': module.v.clone(dev_strs).at_devs()}), len(dev_strs))
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
    x = ivy.DistributedItem([x0, x1])

    # module for processes
    module = TrainableModule(input_channels, output_channels, dev_str=dev_str0, store_vars=False)

    # optimizer
    optim = ivy.SGD(1e-4)

    # return fn
    ret_fn = lambda ret: ivy.unify_iter(ret, dev_str0, 'mean')

    # device manager
    dev_mapper = ivy.DevMapperMultiProc(map_fn, ret_fn, dev_strs, constant={'module': module})

    # local module
    module = TrainableModule(input_channels, output_channels, dev_str=dev_str0, store_vars=True)
    module.build()

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = dev_mapper.map(xn=x.at_devs(), vc=module.v.clone(dev_strs).at_devs())
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


# to_ivy_module_distributed
@pytest.mark.parametrize(
    "bs_ic_oc", [([2, 1], 4, 5)])
@pytest.mark.parametrize(
    "from_class_and_args", [True, False])
@pytest.mark.parametrize(
    "inplace_update", [True, False])
def test_to_ivy_module_distributed(bs_ic_oc, from_class_and_args, inplace_update, dev_str, call):
    # smoke test
    if call is not helpers.torch_call:
        # Currently only implemented for PyTorch
        pytest.skip()

    # devices and inputs
    dev_strs = list()
    xs = list()

    # first device
    dev_str0 = dev_str
    dev_strs.append(dev_str0)

    # first input
    batch_shape, input_channels, output_channels = bs_ic_oc
    dev_batch_shape = [int(batch_shape[0]/2)] + batch_shape[1:]
    xs.append(ivy.cast(ivy.linspace(ivy.zeros(dev_batch_shape), ivy.ones(dev_batch_shape),
                                   input_channels, dev_str=dev_str0), 'float32'))

    # second device
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
        dev_strs.append(dev_str1)

        # second input
        xs.append(ivy.cast(ivy.linspace(ivy.zeros(dev_batch_shape), ivy.ones(dev_batch_shape),
                                       input_channels, dev_str=dev_str1), 'float32'))

    # combined inputs
    x = ivy.DistributedItem(xs)

    # ivy module
    natvie_module_class = NATIVE_MODULES[ivy.current_framework_str()]
    if from_class_and_args:
        ivy_module = ivy.to_ivy_module(native_module_class=natvie_module_class,
                                       args=[input_channels, output_channels],
                                       dev_strs=dev_strs, inplace_update=inplace_update)
    else:
        native_module = natvie_module_class(input_channels, output_channels)
        ivy_module = ivy.to_ivy_module(native_module, dev_strs=dev_strs, inplace_update=inplace_update)

    # optimizer
    optim = ivy.SGD(1e-4)

    # return fn
    ret_fn = lambda ret: ivy.unify_iter(ret, dev_str0, 'mean')

    # test loss_fn
    ret_val = ivy.map(loss_fn,
                      constant={'module': ivy_module},
                      unique={'x_': x.at_devs(),
                              'v_': ivy_module.v.clone(dev_strs).at_devs()})[0]
    assert ivy.is_array(ret_val)

    if inplace_update:
        # inplace_update mode does not support gradient propagation
        return

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss_n_grads = ivy.MultiDevIter(
            ivy.map(map_fn,
                    constant={'module': ivy_module},
                    unique={'dev_str': dev_strs,
                            'xn': x.at_devs(),
                            'vc': ivy_module.v.clone(dev_strs).at_devs()}),
            len(dev_strs))
        loss, grads = ret_fn(loss_n_grads)
        ivy_module.v = optim.step(ivy_module.v, grads)
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


# to_ivy_module_distributed
@pytest.mark.parametrize(
    "bs_ic_oc", [([2, 1], 4, 5)])
@pytest.mark.parametrize(
    "from_class_and_args", [True, False])
@pytest.mark.parametrize(
    "inplace_update", [True, False])
def test_to_ivy_module_distributed_multiprocess(bs_ic_oc, from_class_and_args, inplace_update, dev_str, call):
    # smoke test
    if call is not helpers.torch_call:
        # Currently only implemented for PyTorch
        pytest.skip()

    # devices and inputs
    dev_strs = list()
    xs = list()

    # first device
    dev_str0 = dev_str
    dev_strs.append(dev_str0)

    # first input
    batch_shape, input_channels, output_channels = bs_ic_oc
    dev_batch_shape = [int(batch_shape[0]/2)] + batch_shape[1:]
    xs.append(ivy.cast(ivy.linspace(ivy.zeros(dev_batch_shape), ivy.ones(dev_batch_shape),
                                   input_channels, dev_str=dev_str0), 'float32'))

    # second device
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
        dev_strs.append(dev_str1)

        # second input
        xs.append(ivy.cast(ivy.linspace(ivy.zeros(dev_batch_shape), ivy.ones(dev_batch_shape),
                                       input_channels, dev_str=dev_str1), 'float32'))

    # combined inputs
    x = ivy.DistributedItem(xs)

    # ivy module
    natvie_module_class = NATIVE_MODULES[ivy.current_framework_str()]
    if from_class_and_args:
        ivy_module = ivy.to_ivy_module(native_module_class=natvie_module_class,
                                       args=[input_channels, output_channels],
                                       dev_strs=dev_strs, inplace_update=False)
    else:
        native_module = natvie_module_class(input_channels, output_channels)
        ivy_module = ivy.to_ivy_module(native_module, dev_strs=dev_strs, inplace_update=False)

    # optimizer
    optim = ivy.SGD(1e-4)

    # return fn
    ret_fn = lambda ret: ivy.unify_iter(ret, dev_str0, 'mean')

    # test loss_fn
    ret_val = ivy.map(loss_fn,
                      constant={'module': ivy_module},
                      unique={'x_': x.at_devs(),
                              'v_': ivy_module.v.clone(dev_strs).at_devs()})[0]
    assert ivy.is_array(ret_val)

    if inplace_update:
        # inplace_update mode does not support gradient propagation
        return

    # device manager
    dev_mapper = ivy.DevMapperMultiProc(map_fn, ret_fn, dev_strs, constant={'module': ivy_module})

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = dev_mapper.map(xn=x.at_devs(), vc=ivy_module.v.clone(dev_strs).at_devs())
        ivy_module.v = optim.step(ivy_module.v, grads)
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
