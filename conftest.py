# global
import pytest
from typing import Dict

# local
import ivy
from ivy_tests.test_ivy import helpers


FW_STRS = ["numpy", "jax", "tensorflow", "torch"]


TEST_FRAMEWORKS: Dict[str, callable] = {
    "numpy": lambda: helpers.get_ivy_numpy(),
    "jax": lambda: helpers.get_ivy_jax(),
    "tensorflow": lambda: helpers.get_ivy_tensorflow(),
    "torch": lambda: helpers.get_ivy_torch(),
}


@pytest.fixture(autouse=True)
def run_around_tests(device, f, wrapped_mode, compile_graph, fw):
    if wrapped_mode and fw == "tensorflow_graph":
        # ToDo: add support for wrapped_mode and tensorflow compilation
        pytest.skip()
    if wrapped_mode and fw == "jax":
        # ToDo: add support for wrapped_mode with jax, presumably some errenously wrapped jax methods
        pytest.skip()
    if "gpu" in device and fw == "numpy":
        # Numpy does not support GPU
        pytest.skip()
    ivy.clear_backend_stack()
    with f.use:
        ivy.set_default_device(device)
        yield


def pytest_generate_tests(metafunc):

    # dev_str
    raw_value = metafunc.config.getoption("--device")
    if raw_value == "all":
        devices = ["cpu", "gpu:0", "tpu:0"]
    else:
        devices = raw_value.split(",")

    # framework
    raw_value = metafunc.config.getoption("--framework")
    if raw_value == "all":
        f_strs = TEST_FRAMEWORKS.keys()
    else:
        f_strs = raw_value.split(",")

    # wrapped_mode
    raw_value = metafunc.config.getoption("--wrapped_mode")
    if raw_value == "both":
        wrapped_modes = [True, False]
    elif raw_value == "true":
        wrapped_modes = [True]
    else:
        wrapped_modes = [False]

    # compile_graph
    raw_value = metafunc.config.getoption("--compile_graph")
    if raw_value == "both":
        compile_modes = [True, False]
    elif raw_value == "true":
        compile_modes = [True]
    else:
        compile_modes = [False]

    # create test configs
    configs = list()
    for f_str in f_strs:
        for device in devices:
            for wrapped_mode in wrapped_modes:
                for compile_graph in compile_modes:
                    configs.append(
                        (device, TEST_FRAMEWORKS[f_str](), wrapped_mode, compile_graph, f_str)
                    )
    metafunc.parametrize("device,f,wrapped_mode,compile_graph,fw", configs)


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption(
        "--framework", action="store", default="numpy,jax,tensorflow,torch"
    )
    parser.addoption("--wrapped_mode", action="store", default="false")
    parser.addoption("--compile_graph", action="store", default="true")
