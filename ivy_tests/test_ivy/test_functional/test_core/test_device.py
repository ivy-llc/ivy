"""Collection of tests for unified device functions."""

# global
import os
import math
import pytest
import time
import numpy as np
from numbers import Number
from hypothesis import strategies as st, given

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# Tests #
# ------#

# Device Queries #

# dev


@given(
    array_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
)
def test_dev(array_shape, dtype, as_variable, fw, device):
    if fw == "torch" and "int" in dtype:
        return

    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)
    x = ivy.asarray(x)
    if as_variable:
        x = ivy.variable(x)

    ret = ivy.dev(x)
    # type test
    assert isinstance(ret, str)
    # value test
    assert ret == device


# as_ivy_dev
@given(
    array_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
)
def test_as_ivy_dev(array_shape, dtype, as_variable, fw, device):
    if fw == "torch" and "int" in dtype:
        return

    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)
    x = ivy.asarray(x)
    if as_variable:
        x = ivy.variable(x)

    if (isinstance(x, Number) or x.size == 0) and as_variable and fw == "mxnet":
        # mxnet does not support 0-dimensional variables
        return

    device = ivy.dev(x)
    ret = ivy.as_ivy_dev(device)
    # type test
    assert isinstance(ret, str)


# as_native_dev
@given(
    array_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
)
def test_as_native_dev(array_shape, dtype, as_variable, device, fw, call):
    if fw == "torch" and "int" in dtype:
        return

    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)
    x = ivy.asarray(x)
    if as_variable:
        x = ivy.variable(x)

    if (isinstance(x, Number) or x.size == 0) and as_variable and fw == "mxnet":
        # mxnet does not support 0-dimensional variables
        return

    device = ivy.as_native_dev(device)
    ret = ivy.as_native_dev(ivy.dev(x))
    # value test
    if call in [helpers.tf_call, helpers.tf_graph_call]:
        assert "/" + ":".join(ret[1:].split(":")[-2:]) == "/" + ":".join(
            device[1:].split(":")[-2:]
        )
    elif call is helpers.torch_call:
        assert ret.type == device.type
    else:
        assert ret == device
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not handle converting string to device
        return


# memory_on_dev
@pytest.mark.parametrize("dev_to_check", ["cpu", "gpu:0"])
def test_memory_on_dev(dev_to_check, device, call):
    if "gpu" in dev_to_check and ivy.num_gpus() == 0:
        # cannot get amount of memory for gpu which is not present
        pytest.skip()
    ret = ivy.total_mem_on_dev(dev_to_check)
    # type test
    assert isinstance(ret, float)
    # value test
    assert 0 < ret < 64
    # compilation test
    if call is helpers.torch_call:
        # global variables aren't supported for pytorch scripting
        pytest.skip()


# Device Allocation #

# default_device
def test_default_device(device, call):
    # setting and unsetting
    orig_len = len(ivy.default_device_stack)
    ivy.set_default_device("cpu")
    assert len(ivy.default_device_stack) == orig_len + 1
    ivy.set_default_device("cpu")
    assert len(ivy.default_device_stack) == orig_len + 2
    ivy.unset_default_device()
    assert len(ivy.default_device_stack) == orig_len + 1
    ivy.unset_default_device()
    assert len(ivy.default_device_stack) == orig_len

    # with
    assert len(ivy.default_device_stack) == orig_len
    with ivy.DefaultDevice("cpu"):
        assert len(ivy.default_device_stack) == orig_len + 1
        with ivy.DefaultDevice("cpu"):
            assert len(ivy.default_device_stack) == orig_len + 2
        assert len(ivy.default_device_stack) == orig_len + 1
    assert len(ivy.default_device_stack) == orig_len


# to_dev
@given(
    array_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
)
def test_to_dev(array_shape, dtype, as_variable, with_out, fw, device, call):
    if fw == "torch" and "int" in dtype:
        return

    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)
    x = ivy.asarray(x)
    if as_variable:
        x = ivy.variable(x)

    if (isinstance(x, Number) or x.size == 0) and as_variable and fw == "mxnet":
        # mxnet does not support 0-dimensional variables
        return

    # create a dummy array for out that is broadcastable to x
    out = ivy.zeros(ivy.shape(x)) if with_out else None

    device = ivy.dev(x)
    x_on_dev = ivy.to_dev(x, device=device, out=out)
    dev_from_new_x = ivy.dev(x_on_dev)

    if with_out:
        # should be the same array test
        assert np.allclose(ivy.to_numpy(x_on_dev), ivy.to_numpy(out))

        # should be the same device
        assert ivy.dev(x_on_dev, as_native=True) == ivy.dev(out, as_native=True)

        # check if native arrays are the same
        if ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            return

        assert x_on_dev.data is out.data

    # value test
    if call in [helpers.tf_call, helpers.tf_graph_call]:
        assert "/" + ":".join(dev_from_new_x[1:].split(":")[-2:]) == "/" + ":".join(
            device[1:].split(":")[-2:]
        )
    elif call is helpers.torch_call:
        assert type(dev_from_new_x) == type(device)
    else:
        assert dev_from_new_x == device


# Function Splitting #


@given(
    array_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    chunk_size=st.integers(1, 3),
    axis=st.integers(0, 1),
)
def test_split_func_call(
    array_shape, dtype, as_variable, chunk_size, axis, fw, device, call
):
    if fw == "torch" and "int" in dtype:
        return

    # inputs
    shape = tuple(array_shape)
    x1 = np.random.uniform(size=shape).astype(dtype)
    x2 = np.random.uniform(size=shape).astype(dtype)
    x1 = ivy.asarray(x1)
    x2 = ivy.asarray(x2)
    if as_variable:
        x1 = ivy.variable(x1)
        x1 = ivy.variable(x2)

    # function
    def func(t0, t1):
        return t0 * t1, t0 - t1, t1 - t0

    # predictions
    a, b, c = ivy.split_func_call(
        func, [x1, x2], "concat", chunk_size=chunk_size, input_axes=axis
    )

    # true
    a_true, b_true, c_true = func(x1, x2)

    # value test
    assert np.allclose(ivy.to_numpy(a), ivy.to_numpy(a_true))
    assert np.allclose(ivy.to_numpy(b), ivy.to_numpy(b_true))
    assert np.allclose(ivy.to_numpy(c), ivy.to_numpy(c_true))


@given(
    array_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[2, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    chunk_size=st.integers(1, 3),
    axis=st.integers(0, 1),
)
def test_split_func_call_with_cont_input(
    array_shape, dtype, as_variable, chunk_size, axis, fw, device, call
):
    # Skipping some dtype for certain frameworks
    if (
        (fw == "torch" and "int" in dtype)
        or (fw == "numpy" and "float16" in dtype)
        or (fw == "tensorflow" and "u" in dtype)
    ):
        return
    shape = tuple(array_shape)
    x1 = np.random.uniform(size=shape).astype(dtype)
    x2 = np.random.uniform(size=shape).astype(dtype)
    x1 = ivy.asarray(x1, device=device)
    x2 = ivy.asarray(x2, device=device)
    # inputs

    if as_variable:
        in0 = ivy.Container(cont_key=ivy.variable(x1))
        in1 = ivy.Container(cont_key=ivy.variable(x2))
    else:
        in0 = ivy.Container(cont_key=x1)
        in1 = ivy.Container(cont_key=x2)

    # function
    def func(t0, t1):
        return t0 * t1, t0 - t1, t1 - t0

    # predictions
    a, b, c = ivy.split_func_call(
        func, [in0, in1], "concat", chunk_size=chunk_size, input_axes=axis
    )

    # true
    a_true, b_true, c_true = func(in0, in1)

    # value test
    assert np.allclose(ivy.to_numpy(a.cont_key), ivy.to_numpy(a_true.cont_key))
    assert np.allclose(ivy.to_numpy(b.cont_key), ivy.to_numpy(b_true.cont_key))
    assert np.allclose(ivy.to_numpy(c.cont_key), ivy.to_numpy(c_true.cont_key))


@given(
    array_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[2, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    axis=st.integers(0, 1),
    devs_as_dict=st.booleans(),
)
def test_dist_array(
    array_shape, dtype, as_variable, axis, devs_as_dict, fw, device, call
):
    if fw == "torch" and "int" in dtype:
        return
    # inputs
    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)
    x = ivy.asarray(x)
    if as_variable:
        x = ivy.variable(x)

    # devices
    devices = list()
    dev0 = device
    devices.append(dev0)

    if "gpu" in device and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev1 = device[:-1] + str(idx)
        devices.append(dev1)
    if devs_as_dict:
        devices = dict(
            zip(devices, [int((1 / len(devices)) * x.shape[axis])] * len(devices))
        )

    # return
    x_split = ivy.dev_dist_array(x, devices, axis)

    # shape test
    assert x_split[dev0].shape[axis] == math.floor(x.shape[axis] / len(devices))

    # value test
    assert min([ivy.dev(x_sub) == ds for ds, x_sub in x_split.items()])


@given(
    array_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[2, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    axis=st.integers(0, 1),
)
def test_clone_array(array_shape, dtype, as_variable, axis, fw, device, call):
    if fw == "torch" and "int" in dtype:
        return
    # inputs
    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)
    x = ivy.asarray(x)
    if as_variable:
        x = ivy.variable(x)

    # devices
    devices = list()
    dev0 = device
    devices.append(dev0)
    if "gpu" in device and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev1 = device[:-1] + str(idx)
        devices.append(dev1)

    # return
    x_split = ivy.dev_clone_array(x, devices)

    # shape test
    assert x_split[dev0].shape[axis] == math.floor(x.shape[axis] / len(devices))

    # value test
    assert min([ivy.dev(x_sub) == ds for ds, x_sub in x_split.items()])


@given(
    array_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[2, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
)
def test_unify_array(array_shape, dtype, as_variable, fw, device, call):
    axis = 0
    # TODO: generalise axis
    if fw == "torch" and "int" in dtype:
        return
    # inputs
    xs = np.random.uniform(size=tuple(array_shape)).astype(dtype)

    # devices and inputs
    devices = list()
    dev0 = device
    if as_variable:
        x = {dev0: ivy.variable(ivy.asarray(xs[0], device=dev0))}
    else:
        x = {dev0: ivy.asarray(xs[0], device=dev0)}
    devices.append(dev0)
    if "gpu" in device and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev1 = device[:-1] + str(idx)
        if as_variable:
            x[dev1] = {dev0: ivy.variable(ivy.asarray(xs[1], device=dev0))}
        else:
            x[dev1] = {dev0: ivy.asarray(xs[1], device=dev0)}
        devices.append(dev1)

    # output
    x_unified = ivy.dev_unify_array(
        ivy.DevDistItem(x), device=dev0, mode="concat", axis=axis
    )

    # shape test
    expected_size = 0
    for ds in devices:
        expected_size += x[ds].shape[axis]
    assert x_unified.shape[axis] == expected_size

    # value test
    assert ivy.dev(x_unified) == dev0


@pytest.mark.parametrize("args", [[[0, 1, 2, 3, 4], "some_str", ([1, 2])]])
@pytest.mark.parametrize("kwargs", [{"a": [0, 1, 2, 3, 4], "b": "another_str"}])
@pytest.mark.parametrize("axis", [0])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_dist_nest(args, kwargs, axis, tensor_fn, device, call):
    # inputs
    args = [tensor_fn(args[0], dtype="float32", device=device)] + args[1:]
    kwargs = {"a": tensor_fn(kwargs["a"], dtype="float32", device=device), "b": kwargs["b"]}

    # devices
    devices = list()
    dev0 = device
    devices.append(dev0)
    if "gpu" in device and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev1 = device[:-1] + str(idx)
        devices.append(dev1)

    # returns
    dist_args, dist_kwargs = ivy.dev_dist_nest(args, kwargs, devices, axis=axis)

    # device specific args
    for ds in devices:
        assert dist_args.at_dev(ds)
        assert dist_kwargs.at_dev(ds)

    # value test
    assert min(
        [
            ivy.dev(dist_args_ds[0]) == ds
            for ds, dist_args_ds in dist_args.at_devs().items()
        ]
    )
    assert min(
        [
            ivy.dev(dist_kwargs_ds["a"]) == ds
            for ds, dist_kwargs_ds in dist_kwargs.at_devs().items()
        ]
    )


@pytest.mark.parametrize("args", [[[0, 1, 2, 3, 4], "some_str", ([1, 2])]])
@pytest.mark.parametrize("kwargs", [{"a": [0, 1, 2, 3, 4], "b": "another_str"}])
@pytest.mark.parametrize("axis", [0])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_clone_nest(args, kwargs, axis, tensor_fn, device, call):
    # inputs
    args = [tensor_fn(args[0], "float32", device)] + args[1:]
    kwargs = {"a": tensor_fn(kwargs["a"], "float32", device), "b": kwargs["b"]}

    # devices
    devices = list()
    dev0 = device
    devices.append(dev0)
    if "gpu" in device and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev1 = device[:-1] + str(idx)
        devices.append(dev1)

    # returns
    cloned_args, cloned_kwargs = ivy.dev_clone_nest(args, kwargs, devices)

    # device specific args
    for ds in devices:
        assert cloned_args.at_dev(ds)
        assert cloned_kwargs.at_dev(ds)

    # value test
    assert min(
        [
            ivy.dev(dist_args_ds[0]) == ds
            for ds, dist_args_ds in cloned_args.at_devs().items()
        ]
    )
    assert min(
        [
            ivy.dev(dist_kwargs_ds["a"]) == ds
            for ds, dist_kwargs_ds in cloned_kwargs.at_devs().items()
        ]
    )


@pytest.mark.parametrize("args", [[[[0, 1, 2], [3, 4]], "some_str", ([1, 2])]])
@pytest.mark.parametrize("kwargs", [{"a": [[0, 1, 2], [3, 4]], "b": "another_str"}])
@pytest.mark.parametrize("axis", [0])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_unify_nest(args, kwargs, axis, tensor_fn, device, call):
    # devices
    devices = list()
    dev0 = device
    devices.append(dev0)
    args_dict = dict()
    args_dict[dev0] = tensor_fn(args[0][0], "float32", dev0)
    kwargs_dict = dict()
    kwargs_dict[dev0] = tensor_fn(kwargs["a"][0], "float32", dev0)
    if "gpu" in device and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev1 = device[:-1] + str(idx)
        devices.append(dev1)
        args_dict[dev1] = tensor_fn(args[0][1], "float32", dev1)
        kwargs_dict[dev1] = tensor_fn(kwargs["a"][1], "float32", dev1)

        # inputs
    args = ivy.DevDistNest([ivy.DevDistItem(args_dict)] + args[1:], devices)
    kwargs = ivy.DevDistNest(
        {"a": ivy.DevDistItem(kwargs_dict), "b": kwargs["b"]}, devices
    )

    # outputs
    args_uni, kwargs_uni = ivy.dev_unify_nest(args, kwargs, dev0, "concat", axis=axis)

    # shape test
    expected_size_arg = 0
    expected_size_kwarg = 0
    for ds in devices:
        expected_size_arg += args._data[0][ds].shape[axis]
        expected_size_kwarg += kwargs._data["a"][ds].shape[axis]
    assert args_uni[0].shape[axis] == expected_size_arg
    assert kwargs_uni["a"].shape[axis] == expected_size_kwarg

    # value test
    assert ivy.dev(args_uni[0]) == dev0
    assert ivy.dev(kwargs_uni["a"]) == dev0


# profiler
def test_profiler(device, call):
    # ToDo: find way to prevent this test from hanging when run
    #  alongside other tests in parallel

    # log dir
    this_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(this_dir, "../log")

    # with statement
    with ivy.Profiler(log_dir):
        a = ivy.ones([10])
        b = ivy.zeros([10])
        a + b
    if call is helpers.mx_call:
        time.sleep(1)  # required by MXNet for some reason

    # start and stop methods
    profiler = ivy.Profiler(log_dir)
    profiler.start()
    a = ivy.ones([10])
    b = ivy.zeros([10])
    a + b
    profiler.stop()
    if call is helpers.mx_call:
        time.sleep(1)  # required by MXNet for some reason


@given(num=st.integers(0, 5))
def test_num_arrays_on_dev(num, device):
    arrays = [ivy.array(np.random.uniform(size=2)) for _ in range(num)]
    assert ivy.num_ivy_arrays_on_dev(device) == num
    # to satisfy lint requirements
    del arrays


@given(num=st.integers(0, 5))
def test_get_all_arrays_on_dev(num, device):
    arrays = [ivy.array(np.random.uniform(size=2)) for _ in range(num)]
    arrs_on_dev = list(ivy.get_all_ivy_arrays_on_dev(device).values())
    for arr in arrays:
        assert arr in arrs_on_dev


# Still to Add #
# ---------------#


# print_all_arrays_on_dev
# clear_mem_on_dev
# total_mem_on_dev
# used_mem_on_dev
# percent_used_mem_on_dev
# dev_util
# gpu_is_available
# num_cpu_cores
# num_gpus
# tpu_is_available
# _assert_dev_correct_formatting
# set_default_device
# unset_default_device
# split_factor
# set_split_factor
# isinstance
# Class MultiDev
# class MultiDevItem
# class MultiDevIter
# class MultiDevNest
# class DevDistItem
# class DevDistIter
# class DevDistNest
# class DevClonedItem
# class DevClonedIter
# class DevClonedNest
# dev_clone
# dev_clone_iter
# _concat_unify_array
# _sum_unify_array
# _mean_unify_array
# dev_unify_array
# dev_unify
# dev_unify_iter
# class DevMapper
# class DevMapperMultiProc
# class DevManager
# class Profiler
