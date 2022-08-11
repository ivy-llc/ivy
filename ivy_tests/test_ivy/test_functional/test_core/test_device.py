"""Collection of tests for unified device functions."""

# global
import io
import multiprocessing
import os
import re
import shutil
import sys

import numpy as np
import nvidia_smi
import psutil
import pytest
from hypothesis import strategies as st, given, assume

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Helpers #
# ------- #


def _ram_array_and_clear_test(metric_fn, size=10000000):
    # This function checks if the memory usage changes before, during and after

    # Measure usage before creating array
    before = metric_fn()
    # Create an array of floats, by default with 10 million elements (40 MB)
    arr = ivy.ones((size,), dtype="float32", device="cpu")
    during = metric_fn()
    # Check that the memory usage has increased
    assert before < during

    # Delete the array
    del arr
    # Measure the memory usage after the array is deleted
    after = metric_fn()
    # Check that the memory usage has decreased
    assert during > after


def _get_possible_devices():
    # Return all the possible usable devices
    devices = ["cpu"]
    if ivy.gpu_is_available():
        for i in range(ivy.num_gpus()):
            devices.append("gpu:" + str(i))

    # Return a list of ivy devices
    return list(map(ivy.Device, devices))


def _empty_dir(path, recreate=False):
    # Delete the directory if it exists and create it again if recreate is True
    if os.path.exists(path):
        shutil.rmtree(path)
    if recreate:
        os.makedirs(path)


# Tests #
# ------#

# Device Queries #

# dev


@given(
    array_shape=helpers.lists(
        arg=helpers.ints(min_value=2, max_value=3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
)
@handle_cmd_line_args
def test_dev(*, array_shape, dtype, as_variable, fw):

    assume(not (fw == "torch" and "int" in dtype))
    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)

    for device in _get_possible_devices():
        x = ivy.array(x, device=device)
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
        arg=helpers.ints(min_value=2, max_value=3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
)
@handle_cmd_line_args
def test_as_ivy_dev(*, array_shape, dtype, as_variable, fw):

    assume(not (fw == "torch" and "int" in dtype))

    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)

    for device in _get_possible_devices():
        x = ivy.array(x, device=device)
        if as_variable:
            x = ivy.variable(x)

        native_device = ivy.dev(x, True)
        ret = ivy.as_ivy_dev(native_device)

        # Type test
        assert isinstance(ret, str)
        # Value test
        assert ret == device


# as_native_dev
@given(
    array_shape=helpers.lists(
        arg=helpers.ints(min_value=1, max_value=3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes[1:]),
    data=st.data(),
)
@handle_cmd_line_args
def test_as_native_dev(*, array_shape, dtype, as_variable, fw, call):

    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)

    for device in _get_possible_devices():
        x = ivy.asarray(x, device=device)
        if as_variable:
            x = ivy.variable(x)

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


# memory_on_dev
def test_memory_on_dev(call):
    for device in _get_possible_devices():
        ret = ivy.total_mem_on_dev(device)
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
def test_default_device(device):
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
        arg=helpers.ints(min_value=1, max_value=3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    stream=helpers.ints(min_value=0, max_value=50),
    data=st.data(),
)
@handle_cmd_line_args
def test_to_device(
    *, array_shape, dtype, as_variable, with_out, fw, device, call, stream
):
    assume(not (fw == "torch" and "int" in dtype))

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
        # these backends do not support native inplace updates
        assume(not (fw in ["tensorflow", "jax"]))

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
    max_val = draw(st.shared(helpers.ints(), key="num_dims"))
    return draw(helpers.ints(min_value=0, max_value=max_val - 1))


@given(
    array_shape=helpers.lists(
        arg=helpers.ints(min_value=1, max_value=3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    chunk_size=helpers.ints(min_value=1, max_value=3),
    axis=_axis(),
    data=st.data(),
)
@handle_cmd_line_args
def test_split_func_call(
    *, array_shape, dtype, as_variable, chunk_size, axis, fw, device, call
):
    assume(not (fw == "torch" and "int" in dtype))

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
        arg=helpers.ints(min_value=2, max_value=3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 3],
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    chunk_size=helpers.ints(min_value=1, max_value=3),
    axis=helpers.ints(min_value=0, max_value=1),
    data=st.data(),
)
@handle_cmd_line_args
def test_split_func_call_with_cont_input(
    *, array_shape, dtype, as_variable, chunk_size, axis, fw, device, call
):
    # Skipping some dtype for certain frameworks
    assume(
        not (
            (fw == "torch" and "int" in dtype)
            or (fw == "numpy" and "float16" in dtype)
            or (fw == "tensorflow" and "u" in dtype)
        )
    )

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


# profiler
def test_profiler(device, fw):
    # ToDo: find way to prevent this test from hanging when run
    #  alongside other tests in parallel

    # log dir, each framework uses their own folder,
    # so we can run this test in parallel
    this_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(this_dir, "../log")
    fw_log_dir = os.path.join(log_dir, fw)

    # Remove old content and recreate log dir
    _empty_dir(fw_log_dir, True)

    # with statement
    with ivy.Profiler(fw_log_dir):
        a = ivy.ones([10])
        b = ivy.zeros([10])
        _ = a + b

    # Should have content in folder
    assert len(os.listdir(fw_log_dir)) != 0, "Profiler did not log anything"

    # Remove old content and recreate log dir
    _empty_dir(fw_log_dir, True)

    # Profiler should stop log
    assert len(os.listdir(fw_log_dir)) == 0, "Profiler logged something while stopped"

    # start and stop methods
    profiler = ivy.Profiler(fw_log_dir)
    profiler.start()
    a = ivy.ones([10])
    b = ivy.zeros([10])
    _ = a + b
    profiler.stop()

    # Should have content in folder
    assert len(os.listdir(fw_log_dir)) != 0, "Profiler did not log anything"

    # Remove old content including the logging folder
    _empty_dir(fw_log_dir, False)

    assert not os.path.exists(fw_log_dir), "Profiler recreated logging folder"


@given(num=helpers.ints(min_value=0, max_value=5))
def test_num_arrays_on_dev(num, device):
    arrays = [
        ivy.array(np.random.uniform(size=2).tolist(), device=device) for _ in range(num)
    ]
    assert ivy.num_ivy_arrays_on_dev(device) == num
    for item in arrays:
        del item


@given(num=helpers.ints(min_value=0, max_value=5))
def test_get_all_arrays_on_dev(num, device):
    arrays = [ivy.array(np.random.uniform(size=2)) for _ in range(num)]
    arr_ids_on_dev = [id(a) for a in ivy.get_all_ivy_arrays_on_dev(device).values()]
    for a in arrays:
        assert id(a) in arr_ids_on_dev


@given(num=helpers.ints(min_value=0, max_value=2), attr_only=st.booleans())
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


def test_used_mem_on_dev():
    devices = _get_possible_devices()

    # Check that there not all memory is used
    for device in devices:
        assert ivy.used_mem_on_dev(device) > 0
        assert ivy.used_mem_on_dev(device) < ivy.total_mem_on_dev(device)

    # Testing if it's detects changes in RAM usage, cannot apply this to GPU, as we can
    # only get the total memory usage of a GPU, not the usage by the program.
    _ram_array_and_clear_test(lambda: ivy.used_mem_on_dev(ivy.Device("cpu"), True))


def test_percent_used_mem_on_dev():
    devices = _get_possible_devices()

    for device in devices:
        used = ivy.percent_used_mem_on_dev(ivy.Device(device))
        assert 0 <= used <= 100

    # Same as test_used_mem_on_dev, but using percent of total memory as metric function
    _ram_array_and_clear_test(
        lambda: ivy.percent_used_mem_on_dev(ivy.Device("cpu"), True)
    )


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


# Still to Add #
# ---------------#


# clear_mem_on_dev
# used_mem_on_dev # working fine for cpu
# percent_used_mem_on_dev # working fine for cpu
# dev_util # working fine for cpu
# tpu_is_available
# _assert_dev_correct_formatting
