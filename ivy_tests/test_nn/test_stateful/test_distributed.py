"""
Collection of tests for Ivy distributed training
"""

# global
import pytest

# local
import ivy
import ivy_tests.helpers as helpers


class TrainableModule(ivy.Module):

    def __init__(self, in_size, out_size, dev_str=None, hidden_size=64):
        self._linear0 = ivy.Linear(in_size, hidden_size, dev_str=dev_str)
        self._linear1 = ivy.Linear(hidden_size, hidden_size, dev_str=dev_str)
        self._linear2 = ivy.Linear(hidden_size, out_size, dev_str=dev_str)
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        x = ivy.expand_dims(x, 0)
        x = ivy.tanh(self._linear0(x))
        x = ivy.tanh(self._linear1(x))
        return ivy.tanh(self._linear2(x))[0]


# module training
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_distributed_training(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()

    # input
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')

    # devices
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    dev_strs = [dev_str0, dev_str1]

    # module
    module_dist = ivy.DistributedModule(TrainableModule, dev_strs, in_size=input_channels, out_size=output_channels)

    # optimizer
    optim_dist = ivy.DistributedOptimizer(ivy.Adam, dev_strs, 1e-4)

    def loss_fn(v_):
        out = module_dist(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, module_dist.v)
        module_dist.v = optim_dist.step(module_dist.v, grads)
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
