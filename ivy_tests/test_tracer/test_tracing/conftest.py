# global
import pytest
from typing import Dict

# local
import ivy
from ivy import unset_backend, DefaultDevice

FW_STRS = ["numpy", "jax", "tensorflow", "torch"]


@pytest.fixture(autouse=True)
def run_around_tests(dev, f):
    unset_backend()
    with ivy.utils.backend.ContextManager(f):
        if "gpu" in dev:
            if ivy.current_backend_str() == "numpy":
                pytest.skip()  # Numpy does not support GPU
            assert ivy.gpu_is_available(), "No GPU available"
        with DefaultDevice(dev):
            yield


def pytest_generate_tests(metafunc):
    # device
    raw_value = metafunc.config.getoption("--dev")
    if raw_value == "all":
        devs = ["cpu", "gpu:0", "tpu:0"]
    else:
        devs = raw_value.split(",")

    # backend
    raw_value = metafunc.config.getoption("--backend")
    if "/" in raw_value:
        f_strs = [raw_value.split("/")[0]]
    elif raw_value == "all":
        f_strs = TEST_BACKENDS.keys()
    else:
        f_strs = raw_value.split(",")

    # create test configs
    configs = list()
    for f_str in f_strs:
        for dev in devs:
            configs.append((dev, f_str))
    metafunc.parametrize("dev,f", configs)
