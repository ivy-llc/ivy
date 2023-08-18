"""Collection of tests for unified device functions."""

# global
import io
import multiprocessing
import os
import re
import shutil
import sys
import warnings

import numpy as np
import psutil
import subprocess
from hypothesis import strategies as st, assume

try:
    import pynvml
except ImportError:
    warnings.warn(
        "pynvml installation was not found in the environment, functionalities"
        " of the Ivy's device module will be limited. Please install pynvml if"
        " you wish to use GPUs with Ivy."
    )
    # nvidia-ml-py (pynvml) is not installed in CPU Dockerfile.

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.globals as test_globals
from ivy_tests.test_ivy.helpers import handle_test, update_backend


# Helpers #
# ------- #


def _ram_array_and_clear_test(metric_fn, device, size=10000000):
    # This function checks if the memory usage changes before, during and after

    # Measure usage before creating array
    before = metric_fn()
    # Create an array of floats, by default with 10 million elements (40 MB)
    arr = ivy.random_normal(shape=(size,), dtype="float32", device=device)
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
    with update_backend(test_globals.CURRENT_BACKEND) as ivy_backend:
        devices = ["cpu"]
        if ivy_backend.gpu_is_available():
            for i in range(ivy_backend.num_gpus()):
                devices.append("gpu:" + str(i))

        # Return a list of ivy devices
        return list(map(ivy_backend.Device, devices))


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
@handle_test(
    fn_tree="functional.ivy.dev",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_dev(*, dtype_and_x, test_flags, backend_fw):
    dtype, x = dtype_and_x
    dtype = dtype[0]
    x = x[0]

    with update_backend(backend_fw) as ivy_backend:
        for device in _get_possible_devices():
            x = ivy_backend.array(x, device=device)
            if test_flags.as_variable and ivy_backend.is_float_dtype(dtype):
                x = ivy_backend.functional.ivy.gradients._variable(x)

            ret = ivy_backend.dev(x)
            # type test
            assert isinstance(ret, str)
            # value test
            assert ret == device
            # array instance test
            assert x.dev() == device
            # container instance test
            container_x = ivy_backend.Container({"a": x})
            assert container_x.dev() == device
            # container static test
            assert ivy_backend.Container.static_dev(container_x) == device


# as_ivy_dev
@handle_test(
    fn_tree="functional.ivy.as_ivy_dev",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_as_ivy_dev(*, dtype_and_x, test_flags, backend_fw):
    dtype, x = dtype_and_x
    dtype = dtype[0]
    x = x[0]

    with update_backend(backend_fw) as ivy_backend:
        for device in _get_possible_devices():
            x = ivy_backend.array(x, device=device)
            if test_flags.as_variable and ivy_backend.is_float_dtype(dtype):
                x = ivy_backend.functional.ivy.gradients._variable(x)

            native_device = ivy_backend.dev(x, as_native=True)
            ret = ivy_backend.as_ivy_dev(native_device)

            # Type test
            assert isinstance(ret, str)
            # Value test
            assert ret == device


# as_native_dev
@handle_test(
    fn_tree="functional.ivy.as_native_dev",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_as_native_dev(*, dtype_and_x, test_flags, on_device, backend_fw):
    dtype, x = dtype_and_x
    dtype = dtype[0]
    x = x[0]

    with update_backend(backend_fw) as ivy_backend:
        for device in _get_possible_devices():
            x = ivy_backend.asarray(x, device=on_device)
            if test_flags.as_variable:
                x = ivy_backend.functional.ivy.gradients._variable(x)

            device = ivy_backend.as_native_dev(on_device)
            ret = ivy_backend.as_native_dev(ivy_backend.dev(x))
            # value test
            if backend_fw == "tensorflow":
                assert "/" + ":".join(ret[1:].split(":")[-2:]) == "/" + ":".join(
                    device[1:].split(":")[-2:]
                )
            elif backend_fw == "torch":
                assert ret.type == device.type
            elif backend_fw == "paddle":
                assert ret._equals(device)
            else:
                assert ret == device


# Device Allocation #
# default_device
@handle_test(fn_tree="functional.ivy.default_device")
def test_default_device(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        # setting and unsetting
        orig_len = len(ivy_backend.default_device_stack)
        ivy_backend.set_default_device("cpu")
        assert len(ivy_backend.default_device_stack) == orig_len + 1
        ivy_backend.set_default_device("cpu")
        assert len(ivy_backend.default_device_stack) == orig_len + 2
        ivy_backend.unset_default_device()
        assert len(ivy_backend.default_device_stack) == orig_len + 1
        ivy_backend.unset_default_device()
        assert len(ivy_backend.default_device_stack) == orig_len

        # with
        assert len(ivy_backend.default_device_stack) == orig_len
        with ivy_backend.DefaultDevice("cpu"):
            assert len(ivy_backend.default_device_stack) == orig_len + 1
            with ivy_backend.DefaultDevice("cpu"):
                assert len(ivy_backend.default_device_stack) == orig_len + 2
            assert len(ivy_backend.default_device_stack) == orig_len + 1
        assert len(ivy_backend.default_device_stack) == orig_len


# to_dev
@handle_test(
    fn_tree="functional.ivy.to_device",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    stream=helpers.ints(min_value=0, max_value=50),
)
def test_to_device(
    *,
    dtype_and_x,
    stream,
    test_flags,
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    dtype = dtype[0]
    x = x[0]

    with update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.asarray(x)
        if test_flags.as_variable and ivy_backend.is_float_dtype(dtype):
            x = ivy_backend.functional.ivy.gradients._variable(x)

        # create a dummy array for out that is broadcastable to x
        out = (
            ivy_backend.zeros(ivy_backend.shape(x), device=on_device, dtype=dtype)
            if test_flags.with_out
            else None
        )

        device = ivy_backend.dev(x)
        x_on_dev = ivy_backend.to_device(x, on_device, stream=stream, out=out)
        dev_from_new_x = ivy_backend.dev(x_on_dev)

        if test_flags.with_out:
            # should be the same array test
            assert x_on_dev is out

            # should be the same device
            if backend_fw != "paddle":
                assert ivy_backend.dev(x_on_dev, as_native=True) == ivy_backend.dev(
                    out, as_native=True
                )
            else:
                assert ivy_backend.dev(x_on_dev, as_native=False) == ivy_backend.dev(
                    out, as_native=False
                )

            # check if native arrays are the same
            # these backends do not support native inplace updates
            assume(not (backend_fw in ["tensorflow", "jax"]))

            assert x_on_dev.data is out.data

        # value test
        if backend_fw == "tensorflow":
            assert "/" + ":".join(dev_from_new_x[1:].split(":")[-2:]) == "/" + ":".join(
                device[1:].split(":")[-2:]
            )
        elif backend_fw == "torch":
            assert type(dev_from_new_x) == type(device)
        else:
            assert dev_from_new_x == device

        # array instance test
        assert x.to_device(device).dev() == device
        # container instance test
        container_x = ivy_backend.Container({"x": x})
        assert container_x.to_device(device).dev() == device
        # container static test
        assert ivy_backend.Container.to_device(container_x, device).dev() == device


# handle_soft_device_variable
@handle_test(
    fn_tree="functional.ivy.handle_soft_device_variable",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_handle_soft_device_variable(*, dtype_and_x, backend_fw):
    dtype, x = dtype_and_x
    dtype = dtype[0]
    with update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.to_device(x[0], "cpu")

        def fn(x, y):
            return ivy_backend.add(x, y)

        for device in _get_possible_devices():
            ivy_backend.set_default_device(device)
            out = ivy_backend.handle_soft_device_variable(x, fn=fn, y=x)

            # check if device shifting is successful
            assert out.device == ivy_backend.default_device()


# Function Splitting #


@st.composite
def _axis(draw):
    max_val = draw(st.shared(helpers.ints(), key="num_dims"))
    return draw(helpers.ints(min_value=0, max_value=max_val - 1))


@handle_test(
    fn_tree="functional.ivy.split_func_call",
    array_shape=helpers.lists(
        x=helpers.ints(min_value=1, max_value=3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    chunk_size=helpers.ints(min_value=1, max_value=3),
    axis=_axis(),
)
def test_split_func_call(
    *,
    array_shape,
    dtype,
    chunk_size,
    axis,
    test_flags,
    backend_fw,
):
    with update_backend(backend_fw) as ivy_backend:
        # inputs
        shape = tuple(array_shape)
        x1 = np.random.uniform(size=shape).astype(dtype[0])
        x2 = np.random.uniform(size=shape).astype(dtype[0])
        x1 = ivy_backend.asarray(x1)
        x2 = ivy_backend.asarray(x2)
        if test_flags.as_variable and ivy_backend.is_float_dtype(dtype[0]):
            x1 = ivy_backend.functional.ivy.gradients._variable(x1)
            x2 = ivy_backend.functional.ivy.gradients._variable(x2)

        # function
        def func(t0, t1):
            return t0 * t1, t0 - t1, t1 - t0

        # predictions
        a, b, c = ivy_backend.split_func_call(
            func, [x1, x2], "concat", chunk_size=chunk_size, input_axes=axis
        )

        # true
        a_true, b_true, c_true = func(x1, x2)

        # value test
        helpers.assert_all_close(
            ivy_backend.to_numpy(a), ivy_backend.to_numpy(a_true), backend=backend_fw
        )
        helpers.assert_all_close(
            ivy_backend.to_numpy(b), ivy_backend.to_numpy(b_true), backend=backend_fw
        )
        helpers.assert_all_close(
            ivy_backend.to_numpy(c), ivy_backend.to_numpy(c_true), backend=backend_fw
        )


@handle_test(
    fn_tree="functional.ivy.split_func_call",
    array_shape=helpers.lists(
        x=helpers.ints(min_value=2, max_value=3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 3],
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    chunk_size=helpers.ints(min_value=1, max_value=3),
    axis=helpers.ints(min_value=0, max_value=1),
)
def test_split_func_call_with_cont_input(
    *,
    array_shape,
    test_flags,
    dtype,
    chunk_size,
    axis,
    on_device,
    backend_fw,
):
    with update_backend(backend_fw) as ivy_backend:
        shape = tuple(array_shape)
        x1 = np.random.uniform(size=shape).astype(dtype[0])
        x2 = np.random.uniform(size=shape).astype(dtype[0])
        x1 = ivy_backend.asarray(x1, device=on_device)
        x2 = ivy_backend.asarray(x2, device=on_device)
        # inputs

        if test_flags.as_variable and ivy_backend.is_float_dtype(dtype[0]):
            _variable_fn = ivy_backend.functional.ivy.gradients._variable
            in0 = ivy_backend.Container(cont_key=_variable_fn(x1))
            in1 = ivy_backend.Container(cont_key=_variable_fn(x2))
        else:
            in0 = ivy_backend.Container(cont_key=x1)
            in1 = ivy_backend.Container(cont_key=x2)

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
        helpers.assert_all_close(
            ivy_backend.to_numpy(a.cont_key),
            ivy_backend.to_numpy(a_true.cont_key),
            backend=backend_fw,
        )
        helpers.assert_all_close(
            ivy_backend.to_numpy(b.cont_key),
            ivy_backend.to_numpy(b_true.cont_key),
            backend=backend_fw,
        )
        helpers.assert_all_close(
            ivy_backend.to_numpy(c.cont_key),
            ivy_backend.to_numpy(c_true.cont_key),
            backend=backend_fw,
        )


# profiler
@handle_test(
    fn_tree="functional.ivy.Profiler",
)
def test_profiler(*, backend_fw):
    # ToDo: find way to prevent this test from hanging when run
    #  alongside other tests in parallel

    # log dir, each framework uses their own folder,
    # so we can run this test in parallel
    with update_backend(backend_fw) as ivy_backend:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(this_dir, "../log")
        fw_log_dir = os.path.join(log_dir, backend_fw)

        # Remove old content and recreate log dir
        _empty_dir(fw_log_dir, True)

        # with statement
        with ivy_backend.Profiler(fw_log_dir):
            a = ivy_backend.ones([10])
            b = ivy_backend.zeros([10])
            _ = a + b

        # Should have content in folder
        assert len(os.listdir(fw_log_dir)) != 0, "Profiler did not log anything"

        # Remove old content and recreate log dir
        _empty_dir(fw_log_dir, True)

        # Profiler should stop log
        assert (
            len(os.listdir(fw_log_dir)) == 0
        ), "Profiler logged something while stopped"

        # start and stop methods
        profiler = ivy_backend.Profiler(fw_log_dir)
        profiler.start()
        a = ivy_backend.ones([10])
        b = ivy_backend.zeros([10])
        _ = a + b
        profiler.stop()

        # Should have content in folder
        assert len(os.listdir(fw_log_dir)) != 0, "Profiler did not log anything"

        # Remove old content including the logging folder
        _empty_dir(fw_log_dir, False)

        assert not os.path.exists(fw_log_dir), "Profiler recreated logging folder"


@handle_test(
    fn_tree="functional.ivy.num_ivy_arrays_on_dev",
    num=helpers.ints(min_value=0, max_value=5),
)
def test_num_ivy_arrays_on_dev(
    *,
    num,
    on_device,
    backend_fw,
):
    with update_backend(backend_fw) as ivy_backend:
        arrays = [
            ivy_backend.array(np.random.uniform(size=2).tolist(), device=on_device)
            for _ in range(num)
        ]
        assert ivy_backend.num_ivy_arrays_on_dev(on_device) == num
        for item in arrays:
            del item


@handle_test(
    fn_tree="functional.ivy.get_all_ivy_arrays_on_dev",
    num=helpers.ints(min_value=0, max_value=5),
)
def test_get_all_ivy_arrays_on_dev(
    *,
    num,
    on_device,
    backend_fw,
):
    with update_backend(backend_fw) as ivy_backend:
        arrays = [ivy_backend.array(np.random.uniform(size=2)) for _ in range(num)]
        arr_ids_on_dev = [
            id(a) for a in ivy_backend.get_all_ivy_arrays_on_dev(on_device).values()
        ]
        for a in arrays:
            assert id(a) in arr_ids_on_dev


@handle_test(
    fn_tree="functional.ivy.print_all_ivy_arrays_on_dev",
    num=helpers.ints(min_value=0, max_value=2),
    attr_only=st.booleans(),
)
def test_print_all_ivy_arrays_on_dev(
    *,
    num,
    attr_only,
    on_device,
    backend_fw,
):
    with update_backend(backend_fw) as ivy_backend:
        arr = [ivy_backend.array(np.random.uniform(size=2)) for _ in range(num)]

        # Flush to avoid artifact
        sys.stdout.flush()
        # temporarily redirect output to a buffer
        captured_output = io.StringIO()
        sys.stdout = captured_output

        ivy_backend.print_all_ivy_arrays_on_dev(device=on_device, attr_only=attr_only)
        # Flush again to make sure all data is printed
        sys.stdout.flush()
        written = captured_output.getvalue().splitlines()
        # restore stdout
        sys.stdout = sys.__stdout__

        # Should have written same number of lines as the number of array in device
        assert len(written) == num

        if attr_only:
            # Check that the attribute are printed are in the format of
            # (ivy.Shape(dim,...), type)
            regex = r"^\(ivy.Shape\((\d+,(\d,\d*)*)\), \'\w*\'\)$"
        else:
            # Check that the arrays are printed are in the format of ivy.array(...)
            regex = r"^ivy\.array\(\[.*\]\)$"

        # Clear the array from device
        for item in arr:
            del item

        # Apply the regex search
        assert all([re.match(regex, line) for line in written])


@handle_test(fn_tree="total_mem_on_dev")
def test_total_mem_on_dev(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        devices = _get_possible_devices()
        for device in devices:
            if "cpu" in device:
                assert (
                    ivy_backend.total_mem_on_dev(device)
                    == psutil.virtual_memory().total / 1e9
                )
            elif "gpu" in device:
                handle = ivy_backend.functional.ivy.device._get_nvml_gpu_handle(device)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                assert ivy_backend.total_mem_on_dev(device) == gpu_mem.total / 1e9


@handle_test(fn_tree="used_mem_on_dev")
def test_used_mem_on_dev(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        devices = _get_possible_devices()

        # Check that there not all memory is used
        for device in devices:
            assert ivy_backend.used_mem_on_dev(device) > 0
            assert ivy_backend.used_mem_on_dev(device) < ivy_backend.total_mem_on_dev(
                device
            )

            _ram_array_and_clear_test(
                lambda: ivy_backend.used_mem_on_dev(device, process_specific=True),
                device=device,
            )


@handle_test(fn_tree="percent_used_mem_on_dev")
def test_percent_used_mem_on_dev(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        devices = _get_possible_devices()

        for device in devices:
            used = ivy_backend.percent_used_mem_on_dev(ivy_backend.Device(device))
            assert 0 <= used <= 100

            # Same as test_used_mem_on_dev, but using percent of total memory as metric
            # function
            _ram_array_and_clear_test(
                lambda: ivy_backend.percent_used_mem_on_dev(
                    device, process_specific=True
                ),
                device=device,
            )


@handle_test(fn_tree="gpu_is_available")
def test_gpu_is_available(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        # If gpu is available but cannot be initialised it will fail the test
        if ivy_backend.gpu_is_available():
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                assert False


@handle_test(fn_tree="num_cpu_cores")
def test_num_cpu_cores(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        # using multiprocessing module too because ivy uses psutil as basis.
        p_cpu_cores = psutil.cpu_count()
        m_cpu_cores = multiprocessing.cpu_count()
        assert type(ivy_backend.num_cpu_cores()) == int
        assert ivy_backend.num_cpu_cores() == p_cpu_cores
        assert ivy_backend.num_cpu_cores() == m_cpu_cores


def _composition_1(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        return ivy_backend.relu().argmax()


def _composition_2(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        return ivy_backend.ceil() or ivy_backend.floor()


# function_unsupported_devices
@handle_test(
    fn_tree="functional.ivy.function_supported_devices",
    func=st.sampled_from([_composition_1, _composition_2]),
    expected=st.just(["cpu"]),
)
def test_function_supported_devices(
    *,
    func,
    expected,
    backend_fw,
):
    with update_backend(backend_fw) as ivy_backend:
        res = ivy_backend.function_supported_devices(func)
        exp = set(expected)

        assert sorted(tuple(exp)) == sorted(res)


# function_unsupported_devices
@handle_test(
    fn_tree="functional.ivy.function_supported_devices",
    func=st.sampled_from([_composition_1, _composition_2]),
    expected=st.just(["gpu", "tpu"]),
)
def test_function_unsupported_devices(
    *,
    func,
    expected,
    backend_fw,
):
    with update_backend(backend_fw) as ivy_backend:
        res = ivy_backend.function_unsupported_devices(func)
        exp = set(expected)

        assert sorted(tuple(exp)) == sorted(res)


def get_gpu_mem_usage(backend, device="gpu:0"):
    handle = backend.ivy.functional.ivy.device._get_nvml_gpu_handle(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return (info.used / info.total) * 100


@handle_test(fn_tree="clear_cached_mem_on_dev")
def test_clear_cached_mem_on_dev(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        devices = _get_possible_devices()
        for device in devices:
            # Testing on only GPU since clearing cache mem is relevant
            # for only CUDA devices
            if "gpu" in device:
                arr = ivy_backend.random_normal(  # noqa: F841
                    shape=(10000, 1000), dtype="float32", device=device
                )
                del arr
                before = get_gpu_mem_usage(device)
                ivy_backend.clear_cached_mem_on_dev(device)
                after = get_gpu_mem_usage(device)
                assert before > after


def get_cpu_percent():
    output = str(subprocess.check_output(["top", "-bn1"]))
    cpu_percent = float(re.search(r"%Cpu\(s\):\s+([\d.]+)\s+us", output).group(1))
    return cpu_percent


@handle_test(fn_tree="dev_util")
def test_dev_util(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        devices = _get_possible_devices()
        for device in devices:
            # The internally called psutil.cpu_percent() has a unique behavior where it
            # returns 0 as usage when run the second time in same line so simple
            # assert psutil.cpu_percent() == ivy.dev_util(device) isn't possible
            if "cpu" in device:
                assert 100 >= ivy_backend.dev_util(device) >= 0
                # Comparing CPU utilization using top. Two percentiles won't be directly
                # equal but absolute difference should be below a safe threshold
                assert abs(get_cpu_percent() - ivy_backend.dev_util(device)) < 10
            elif "gpu" in device:
                handle = ivy_backend.functional.ivy.device._get_nvml_gpu_handle(device)
                assert (
                    ivy_backend.dev_util(device)
                    == pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                )


@handle_test(fn_tree="tpu_is_available")
def test_tpu_is_available(backend_fw):
    with update_backend(backend_fw) as ivy_backend:
        import tensorflow as tf

        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            tf.config.list_logical_devices("TPU")
            tf.distribute.experimental.TPUStrategy(resolver)
            ground_truth = True
        except ValueError:
            ground_truth = False

        assert ivy_backend.tpu_is_available() == ground_truth
