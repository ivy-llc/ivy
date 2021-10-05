"""
Collection of tests for templated device functions
"""

# global
import math
import pytest
import numpy as np
from numbers import Number

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# Tests #
# ------#

# Device Queries #

# dev
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_dev(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.dev(x)
    # type test
    assert isinstance(ret, ivy.Device)
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dev)


# dev_to_str
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_dev_to_str(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    dev = ivy.dev(x)
    ret = ivy.dev_to_str(dev)
    # type test
    assert isinstance(ret, str)
    # value test
    assert ret == dev_str
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dev_to_str)


# str_to_dev
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_str_to_dev(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    dev = ivy.dev(x)
    ret = ivy.str_to_dev(dev_str)
    # value test
    if call in [helpers.tf_call, helpers.tf_graph_call]:
        assert '/' + ':'.join(ret[1:].split(':')[-2:]) == '/' + ':'.join(dev[1:].split(':')[-2:])
    elif call is helpers.torch_call:
        assert ret.type == dev.type
    else:
        assert ret == dev
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not handle converting string to device
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.str_to_dev)


# dev_str
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_dev_str(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.dev_str(x)
    # type test
    assert isinstance(ret, str)
    # value test
    assert ret == dev_str
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dev_str)


# memory_on_dev
@pytest.mark.parametrize(
    "dev_str_to_check", ['cpu', 'cpu:0', 'gpu:0'])
def test_memory_on_dev(dev_str_to_check, dev_str, call):
    if 'gpu' in dev_str_to_check and ivy.num_gpus() == 0:
        # cannot get amount of memory for gpu which is not present
        pytest.skip()
    ret = ivy.memory_on_dev(dev_str_to_check)
    # type test
    assert isinstance(ret, float)
    # value test
    assert 0 < ret < 64
    # compilation test
    if call is helpers.torch_call:
        # global variables aren't supported for pytorch scripting
        pytest.skip()
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.memory_on_dev)


# Device Allocation #

# to_dev
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_to_dev(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    dev = ivy.dev(x)
    x_on_dev = ivy.to_dev(x, dev_str)
    dev_from_new_x = ivy.dev(x)
    # value test
    if call in [helpers.tf_call, helpers.tf_graph_call]:
        assert '/' + ':'.join(dev_from_new_x[1:].split(':')[-2:]) == '/' + ':'.join(dev[1:].split(':')[-2:])
    elif call is helpers.torch_call:
        assert dev_from_new_x.type == dev.type
    else:
        assert dev_from_new_x == dev
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not handle converting string to device
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.to_dev)


# Function Splitting #

@pytest.mark.parametrize(
    "x0", [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
           [[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
@pytest.mark.parametrize(
    "x1", [[[2, 4, 6], [8, 10, 12], [14, 16, 18]],
           [[18, 16, 14], [12, 10, 8], [6, 4, 2]]])
@pytest.mark.parametrize(
    "chunk_size", [1, 3])
@pytest.mark.parametrize(
    "axis", [0, 1])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_split_func_call(x0, x1, chunk_size, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    in0 = tensor_fn(x0, 'float32', dev_str)
    in1 = tensor_fn(x1, 'float32', dev_str)

    # function
    def func(t0, t1):
        return t0 * t1, t0 - t1, t1 - t0

    # predictions
    a, b, c = ivy.split_func_call(func, [in0, in1], chunk_size, axis)

    # true
    a_true, b_true, c_true = func(in0, in1)

    # value test
    assert np.allclose(ivy.to_numpy(a), ivy.to_numpy(a_true))
    assert np.allclose(ivy.to_numpy(b), ivy.to_numpy(b_true))
    assert np.allclose(ivy.to_numpy(c), ivy.to_numpy(c_true))


@pytest.mark.parametrize(
    "x0", [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
           [[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
@pytest.mark.parametrize(
    "x1", [[[2, 4, 6], [8, 10, 12], [14, 16, 18]],
           [[18, 16, 14], [12, 10, 8], [6, 4, 2]]])
@pytest.mark.parametrize(
    "chunk_size", [1, 3])
@pytest.mark.parametrize(
    "axis", [0, 1])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_split_func_call_with_cont_input(x0, x1, chunk_size, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    in0 = ivy.Container(cont_key=tensor_fn(x0, 'float32', dev_str))
    in1 = ivy.Container(cont_key=tensor_fn(x1, 'float32', dev_str))

    # function
    def func(t0, t1):
        return t0 * t1, t0 - t1, t1 - t0

    # predictions
    a, b, c = ivy.split_func_call(func, [in0, in1], chunk_size, axis)

    # true
    a_true, b_true, c_true = func(in0, in1)

    # value test
    assert np.allclose(ivy.to_numpy(a.cont_key), ivy.to_numpy(a_true.cont_key))
    assert np.allclose(ivy.to_numpy(b.cont_key), ivy.to_numpy(b_true.cont_key))
    assert np.allclose(ivy.to_numpy(c.cont_key), ivy.to_numpy(c_true.cont_key))


@pytest.mark.parametrize(
    "x", [[0, 1, 2, 3, 4]])
@pytest.mark.parametrize(
    "axis", [0])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_distribute_array(x, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    x = tensor_fn(x, 'float32', dev_str)

    # predictions
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    dev_strs = [dev_str0, dev_str1]
    x_split = ivy.distribute_array(x, dev_strs, axis)

    # shape test
    assert len(x_split) == math.floor(x.shape[axis] / len(dev_strs))

    # value test
    assert min([ivy.dev_str(x_sub) == dev_strs[i] for i, x_sub in enumerate(x_split.at_devs())])


@pytest.mark.parametrize(
    "x", [[0, 1, 2, 3, 4]])
@pytest.mark.parametrize(
    "axis", [0])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_clone_array(x, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    x = tensor_fn(x, 'float32', dev_str)

    # predictions
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    dev_strs = [dev_str0, dev_str1]
    x_split = ivy.clone_array(x, dev_strs)

    # shape test
    assert len(x_split) == math.floor(x.shape[axis] / len(dev_strs))

    # value test
    assert min([ivy.dev_str(x_sub) == dev_strs[i] for i, x_sub in enumerate(x_split.at_devs())])


@pytest.mark.parametrize(
    "xs", [([0, 1, 2], [3, 4])])
@pytest.mark.parametrize(
    "axis", [0])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_unify_array(xs, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # devices
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str

    # inputs
    x0 = tensor_fn(xs[0], 'float32', dev_str0)
    x1 = tensor_fn(xs[1], 'float32', dev_str1)

    # output
    x_unified = ivy.unify_array(ivy.DistributedItem([x0, x1]), dev_str0, 'concat', axis)

    # shape test
    assert x_unified.shape[axis] == x0.shape[axis] + x1.shape[axis]

    # value test
    assert ivy.dev_str(x_unified) == dev_str0


@pytest.mark.parametrize(
    "args", [[[0, 1, 2, 3, 4], 'some_str', ([1, 2])]])
@pytest.mark.parametrize(
    "kwargs", [{'a': [0, 1, 2, 3, 4], 'b': 'another_str'}])
@pytest.mark.parametrize(
    "axis", [0])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_distribute_args(args, kwargs, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    args = [tensor_fn(args[0], 'float32', dev_str)] + args[1:]
    kwargs = {'a': tensor_fn(kwargs['a'], 'float32', dev_str), 'b': kwargs['b']}

    # predictions
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    dev_strs = [dev_str0, dev_str1]
    dist_args, dist_kwargs = ivy.distribute_nest(args, kwargs, dev_strs, axis=axis)

    # device specific args
    assert dist_args.at_dev(0)
    assert dist_args.at_dev(1)
    three_present = True
    try:
        dist_args.at_dev(2)
    except IndexError:
        three_present = False
    assert not three_present
    assert dist_kwargs.at_dev(0)
    assert dist_kwargs.at_dev(1)
    three_present = True
    try:
        dist_kwargs.at_dev(3)
    except IndexError:
        three_present = False
    assert not three_present

    # value test
    assert min([ivy.dev_str(dist_args_i[0]) == dev_strs[i]
                for i, dist_args_i in enumerate(dist_args.at_devs())])
    assert min([ivy.dev_str(dist_kwargs_i['a']) == dev_strs[i]
                for i, dist_kwargs_i in enumerate(dist_kwargs.at_devs())])


@pytest.mark.parametrize(
    "args", [[[0, 1, 2, 3, 4], 'some_str', ([1, 2])]])
@pytest.mark.parametrize(
    "kwargs", [{'a': [0, 1, 2, 3, 4], 'b': 'another_str'}])
@pytest.mark.parametrize(
    "axis", [0])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_clone_args(args, kwargs, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    args = [tensor_fn(args[0], 'float32', dev_str)] + args[1:]
    kwargs = {'a': tensor_fn(kwargs['a'], 'float32', dev_str), 'b': kwargs['b']}

    # predictions
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    dev_strs = [dev_str0, dev_str1]
    cloned_args, cloned_kwargs = ivy.clone_nest(args, kwargs, dev_strs)

    # device specific args
    assert cloned_args.at_dev(0)
    assert cloned_args.at_dev(1)
    three_present = True
    try:
        cloned_args.at_dev(3)
    except IndexError:
        three_present = False
    assert not three_present
    assert cloned_kwargs.at_dev(0)
    assert cloned_kwargs.at_dev(1)
    three_present = True
    try:
        cloned_kwargs.at_dev(3)
    except IndexError:
        three_present = False
    assert not three_present

    # value test
    assert min([ivy.dev_str(dist_args_i[0]) == dev_strs[i]
                for i, dist_args_i in enumerate(cloned_args.at_devs())])
    assert min([ivy.dev_str(dist_kwargs_i['a']) == dev_strs[i]
                for i, dist_kwargs_i in enumerate(cloned_kwargs.at_devs())])


@pytest.mark.parametrize(
    "args", [[[[0, 1, 2], [3, 4]], 'some_str', ([1, 2])]])
@pytest.mark.parametrize(
    "kwargs", [{'a': [[0, 1, 2], [3, 4]], 'b': 'another_str'}])
@pytest.mark.parametrize(
    "axis", [0])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_unify_args(args, kwargs, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # devices
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    dev_strs = [dev_str0, dev_str1]
    arg_len = len(dev_strs)

    # inputs
    args = ivy.DistributedNest([ivy.DistributedItem([tensor_fn(args[0][0], 'float32', dev_str0),
                                                     tensor_fn(args[0][1], 'float32', dev_str1)])] + args[1:], arg_len)
    kwargs = ivy.DistributedNest({'a': ivy.DistributedItem([tensor_fn(kwargs['a'][0], 'float32', dev_str0),
                                                            tensor_fn(kwargs['a'][1], 'float32', dev_str1)]),
                                  'b': kwargs['b']}, arg_len)

    # outputs
    args_uni, kwargs_uni = ivy.unify_nest(args, kwargs, dev_str0, 'concat', axis=axis)

    # shape test
    assert args_uni[0].shape[axis] == args._iterable[0].at_dev(0).shape[axis] +\
           args._iterable[0].at_dev(1).shape[axis]
    assert kwargs_uni['a'].shape[axis] == kwargs._iterable['a'].at_dev(0).shape[axis] +\
           kwargs._iterable['a'].at_dev(1).shape[axis]

    # value test
    assert ivy.dev_str(args_uni[0]) == dev_str0
    assert ivy.dev_str(kwargs_uni['a']) == dev_str0
