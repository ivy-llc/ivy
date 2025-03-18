# global
import pytest
from typing import Dict

# local
import ivy
from ivy import unset_backend, DefaultDevice

FW_STRS = ["numpy", "jax", "paddle", "tensorflow", "torch"]


@pytest.fixture(autouse=True)
def run_around_tests(dev, f):
    if "gpu" in dev and ivy.current_backend_str() == "numpy":
        # Numpy does not support GPU
        pytest.skip()
    unset_backend()
    with ivy.utils.backend.ContextManager(f):
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
