"""Collection of tests for unified device functions."""
# global
import io
import math
import multiprocessing
import os
import re
import sys
import time
from numbers import Number

import numpy as np
import nvidia_smi
import psutil
import pytest
from hypothesis import strategies as st, given

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# Tests #
# ------#

# Device Queries #

# dev


@given(
    array_shape=helpers.lists(
        arg=st.integers(2, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
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
    # array instance test
    assert x.dev() == device
    # container instance test
    container_x = ivy.Container({"a": x})
    assert container_x.dev() == device
    # container static test
    assert ivy.Container.static_dev(container_x) == device


# as_ivy_dev
@given(
    array_shape=helpers.lists(
        arg=st.integers(2, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
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
        arg=st.integers(1, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
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
        arg=st.integers(1, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    stream=st.integers(0, 50),
)
def test_to_device(array_shape, dtype, as_variable, with_out, fw, device, call, stream):
    if fw == "torch" and "int" in dtype:
        return
    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)
    x = ivy.asarray(x)
    if as_variable:
        x = ivy.variable(x)

    # create a dummy array for out that is broadcastable to x
    out = ivy.zeros(ivy.shape(x), device=device, dtype=dtype) if with_out else None

    device = ivy.dev(x)
    x_on_dev = ivy.to_device(x, device=device, stream=stream, out=out)
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

    # array instance test
    assert x.to_device(device).dev() == device
    # container instance test
    container_x = ivy.Container({"x": x})
    assert container_x.to_device(device).dev() == device
    # container static test
    assert ivy.Container.to_device(container_x, device).dev() == device


# Function Splitting #


@st.composite
def _axis(draw):
    max_val = draw(st.shared(st.integers(), key="num_dims"))
    return draw(st.integers(0, max_val - 1))


@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    chunk_size=st.integers(1, 3),
    axis=_axis(),
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
        x2 = ivy.variable(x2)

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
        arg=st.integers(2, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 3],
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
        arg=st.integers(2, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 3],
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
        arg=st.integers(2, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 3],
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
        arg=st.integers(2, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 3],
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
    kwargs = {
        "a": tensor_fn(kwargs["a"], dtype="float32", device=device),
        "b": kwargs["b"],
    }

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


# @pytest.mark.parametrize("args", [[[0, 1, 2, 3, 4], "some_str", ([1, 2])]])
# @pytest.mark.parametrize("kwargs", [{"a": [0, 1, 2, 3, 4], "b": "another_str"}])
# @pytest.mark.parametrize("axis", [0])
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
@given(
    arr1=st.lists(st.integers(1, 4), min_size=1, max_size=5),
    arr2=st.lists(st.integers(1, 4), min_size=1, max_size=5),
)
def test_clone_nest(arr1, arr2):
    # Test the function by passing the parameter to sum

    # Custom sum function to test the inputs on
    def my_sum(a1, a2):
        return sum(a1) + sum(a2)

    # create an array from shape
    arr1 = ivy.array(arr1)
    arr2 = ivy.array(arr2)

    # inputs
    args = [arr1, arr2]
    kwargs = {"a1": arr1, "a2": arr2}

    # devices
    devices = ["cpu"]
    if ivy.gpu_is_available():
        devices.append(ivy.Device("gpu:0"))

    # returns
    cloned_args, cloned_kwargs = ivy.dev_clone_nest(args, kwargs, devices)

    # device specific args
    for ds in devices:
        assert cloned_args.at_dev(ds)
        assert cloned_kwargs.at_dev(ds)

    # value test
    for ds, dist_args_ds in cloned_args.at_devs().items():
        assert list(map(list, dist_args_ds)) == list(map(list, args))
        assert my_sum(*dist_args_ds) == my_sum(*args)
    for ds, dist_kwargs_ds in cloned_kwargs.at_devs().items():
        assert list(map(list, dist_kwargs_ds)) == list(map(list, kwargs))
        assert my_sum(**dist_kwargs_ds) == my_sum(**kwargs)


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
    args_dict[dev0] = tensor_fn(args[0][0], dtype="float32", device=dev0)
    kwargs_dict = dict()
    kwargs_dict[dev0] = tensor_fn(kwargs["a"][0], dtype="float32", device=dev0)
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
    arrays = [
        ivy.array(np.random.uniform(size=2).tolist(), device=device) for _ in range(num)
    ]
    assert ivy.num_ivy_arrays_on_dev(device) == num
    arrays.clear()


@given(num=st.integers(0, 5))
def test_get_all_arrays_on_dev(num, device):
    arrays = [ivy.array(np.random.uniform(size=2)) for _ in range(num)]
    arr_ids_on_dev = [id(a) for a in ivy.get_all_ivy_arrays_on_dev(device).values()]
    for a in arrays:
        assert id(a) in arr_ids_on_dev


@given(num=st.integers(0, 2), attr_only=st.booleans())
def test_print_all_ivy_arrays_on_dev(num, device, attr_only):
    arr = [ivy.array(np.random.uniform(size=2)) for _ in range(num)]

    # Flush to avoid artifact
    sys.stdout.flush()
    # temporarily redirect output to a buffer
    captured_output = io.StringIO()
    sys.stdout = captured_output

    ivy.print_all_ivy_arrays_on_dev(device, attr_only=attr_only)
    # Flush again to make sure all data is printed
    sys.stdout.flush()
    written = captured_output.getvalue().splitlines()
    # restore stdout
    sys.stdout = sys.__stdout__

    # Should have written same number of lines as the number of array in device
    assert len(written) == num

    if attr_only:
        # Check that the attribute are printed are in the format of ((dim,...), type)
        regex = r"^\(\((\d+,(\d,\d*)*)\), \'\w*\'\)$"
    else:
        # Check that the arrays are printed are in the format of ivy.array(...)
        regex = r"^ivy\.array\(\[.*\]\)$"

    # Clear the array from device
    for item in arr:
        del item

    # Apply the regex search
    assert all([re.match(regex, line) for line in written])


def test_total_mem_on_dev(device):
    if "cpu" in device:
        assert ivy.total_mem_on_dev(device) == psutil.virtual_memory().total / 1e9
    elif "gpu" in device:
        gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(device)
        assert ivy.total_mem_on_dev(device) == gpu_mem / 1e9


def test_gpu_is_available(fw):
    # If gpu is available but cannot be initialised it will fail the test
    if ivy.gpu_is_available():
        try:
            nvidia_smi.nvmlInit()
        except (
            nvidia_smi.NVMLError_LibraryNotFound,
            nvidia_smi.NVMLError_DriverNotLoaded,
        ):
            assert False


def test_num_cpu_cores():
    # using multiprocessing module too because ivy uses psutil as basis.
    p_cpu_cores = psutil.cpu_count()
    m_cpu_cores = multiprocessing.cpu_count()
    assert type(ivy.num_cpu_cores()) == int
    assert ivy.num_cpu_cores() == p_cpu_cores
    assert ivy.num_cpu_cores() == m_cpu_cores


def test_num_gpus():
    # If there is a gpu then use nvidia_smi to check how many
    if ivy.gpu_is_available():
        # Initialise nvidia_smi
        nvidia_smi.nvmlInit()
        gpu_cores = nvidia_smi.nvmlDeviceGetCount()

        # Type check
        assert type(ivy.num_gpus()) == int
        # Value check
        assert ivy.num_gpus() == gpu_cores
    else:
        # Otherwise there must be no gpus
        assert ivy.num_gpus() == 0


@given(arr=st.lists(st.integers(0, 10), min_size=1, max_size=5))
def test_set_unset_default_device(arr):
    has_gpu = ivy.gpu_is_available()

    # If using cpu, try changing to gpu, and vise versa
    if ivy.default_device() == "cpu":
        # Arrays must use the default device if not specified
        assert ivy.array(arr).dev().startswith("cpu")

        # If it has gpu try setting it to gpu, otherwise try setting it to cpu
        ivy.set_default_device(ivy.Device("gpu:0" if has_gpu else "cpu"))
        assert ivy.array(arr).dev().startswith("gpu" if has_gpu else "cpu")

        # unsetting will revert to what it was before
        ivy.unset_default_device()
        assert ivy.array(arr).dev().startswith("cpu")

    else:
        # Arrays must use the default device if not specified
        assert ivy.array(arr).dev().startswith("gpu")

        # Try setting to gpu, and default device should change
        ivy.set_default_device(ivy.Device("cpu"))
        assert ivy.array(arr).dev().startswith("cpu")

        # unsetting will revert to what it was before
        ivy.unset_default_device()
        assert ivy.array(arr).dev().startswith("gpu")


@given(facs=st.lists(st.floats(0, 1), min_size=2, max_size=2).map(tuple))
def test_split_factor(facs):
    (fac1, fac2) = facs
    has_gpu = ivy.gpu_is_available()
    default_device = ivy.default_device()

    # If not configured then default is 0
    assert ivy.split_factor() == 0

    # Setting the split factor should change the value
    ivy.set_split_factor(fac1)
    assert ivy.split_factor() == fac1

    # Setting the split factor with without a device sets the default device
    assert ivy.split_factor(default_device) == fac1

    # Unsetting the split factor for tests
    ivy.set_split_factor(0)

    # Additional test if gpu is available
    if has_gpu:
        other_device = ivy.Device("gpu:0" if default_device == "cpu" else "cpu")

        # The other device defaults to 0 too
        assert ivy.split_factor(other_device) == 0

        # Setting the split factor
        ivy.set_split_factor(fac2, other_device)
        assert ivy.split_factor(other_device) == fac2

        # unsetting the split factor for tests
        ivy.set_split_factor(0, other_device)


@given(arr=st.lists(st.integers(0, 10), min_size=1, max_size=5))
def test_dev_clone_array(arr):
    has_gpu = ivy.gpu_is_available()
    default_device = ivy.default_device()

    ivy_arr = ivy.array(arr)

    # Clone the array
    cloned = ivy.dev_clone_array(ivy_arr, [default_device])

    # Check it did get cloned
    assert default_device in cloned.keys()

    # Check the array is in the correct device
    assert cloned[default_device].dev() == default_device

    # Check the value is the same
    assert list(cloned[default_device]) == list(arr)

    # Check the array instance method
    instance_cloned = ivy_arr.dev_clone_array([default_device])
    assert list(instance_cloned[default_device]) == list(arr)

    # Check the container instance method
    container = ivy.Container({"arr": ivy_arr})
    instance_cloned = container.dev_clone_array([default_device])
    assert list(instance_cloned["arr"][default_device]) == list(arr)

    # Check the container static method
    instance_cloned = ivy.Container.static_dev_clone_array(container, [default_device])
    assert list(instance_cloned["arr"][default_device]) == list(arr)

    # Extra test if gpu is available
    if has_gpu:
        other_device = ivy.Device("gpu:0" if default_device == "cpu" else "cpu")
        multi_cloned = ivy.dev_clone_array(ivy_arr, [default_device, other_device])

        # Check there is 2 arrays that was cloned
        assert len(multi_cloned) == 2

        # Check both array have the right device
        assert multi_cloned[default_device].dev() == default_device
        assert multi_cloned[other_device].dev() == other_device

        # Check both array have the right values
        assert list(multi_cloned[default_device]) == arr
        assert list(multi_cloned[other_device]) == arr

    pass


# Still to Add #
# ---------------#


# clear_mem_on_dev
# used_mem_on_dev # working fine for cpu
# percent_used_mem_on_dev # working fine for cpu
# dev_util # working fine for cpu
# tpu_is_available
# _assert_dev_correct_formatting
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
