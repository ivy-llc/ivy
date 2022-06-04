# global
import os
import pytest
from typing import Dict
from hypothesis import settings

settings.register_profile("default", max_examples=10, deadline=None)
settings.load_profile("default")

# local
from ivy_tests.test_ivy import helpers
from ivy import clear_backend_stack, DefaultDevice


FW_STRS = ["numpy", "jax", "tensorflow", "torch", "mxnet"]


TEST_BACKENDS: Dict[str, callable] = {
    "numpy": lambda: helpers.get_ivy_numpy(),
    "jax": lambda: helpers.get_ivy_jax(),
    "tensorflow": lambda: helpers.get_ivy_tensorflow(),
    "torch": lambda: helpers.get_ivy_torch(),
    "mxnet": lambda: helpers.get_ivy_mxnet(),
}
TEST_CALL_METHODS: Dict[str, callable] = {
    "numpy": helpers.np_call,
    "jax": helpers.jnp_call,
    "tensorflow": helpers.tf_call,
    "torch": helpers.torch_call,
    "mxnet": helpers.mx_call,
}

if "ARRAY_API_TESTS_MODULE" not in os.environ:
    os.environ["ARRAY_API_TESTS_MODULE"] = "ivy.functional.backends.numpy"


@pytest.fixture(autouse=True)
def run_around_tests(device, f, compile_graph, implicit, call, fw):
    if "gpu" in device and call is helpers.np_call:
        # Numpy does not support GPU
        pytest.skip()
    clear_backend_stack()
    with f.use:
        with DefaultDevice(device):
            yield


def pytest_generate_tests(metafunc):

    # device
    raw_value = metafunc.config.getoption("--device")
    if raw_value == "all":
        devices = ["cpu", "gpu:0", "tpu:0"]
    else:
        devices = raw_value.split(",")

    # framework
    raw_value = metafunc.config.getoption("--backend")
    if raw_value == "all":
        backend_strs = TEST_BACKENDS.keys()
    else:
        backend_strs = raw_value.split(",")

    # compile_graph
    raw_value = metafunc.config.getoption("--compile_graph")
    if raw_value == "both":
        compile_modes = [True, False]
    elif raw_value == "true":
        compile_modes = [True]
    else:
        compile_modes = [False]

    # implicit
    raw_value = metafunc.config.getoption("--with_implicit")
    if raw_value == "true":
        implicit_modes = [True, False]
    else:
        implicit_modes = [False]

    # create test configs
    configs = list()
    for backend_str in backend_strs:
        for device in devices:
            for compile_graph in compile_modes:
                for implicit in implicit_modes:
                    configs.append(
                        (
                            device,
                            TEST_BACKENDS[backend_str](),
                            compile_graph,
                            implicit,
                            TEST_CALL_METHODS[backend_str],
                            backend_str,
                        )
                    )
    metafunc.parametrize("device,f,compile_graph,implicit,call,fw", configs)


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("--backend", action="store", default="jax,numpy,tensorflow,torch")
    parser.addoption("--compile_graph", action="store", default="true")
    parser.addoption("--with_implicit", action="store", default="false")
