import ivy
import os
import pytest

TARGET_FRAMEWORKS = ["numpy", "jax", "tensorflow"]
BACKEND_COMPILE = False
TARGET = "all"
NO_LOGS = False


@pytest.fixture(autouse=True)
def run_around_tests():
    ivy.unset_backend()


def pytest_addoption(parser):
    parser.addoption(
        "--backend-compile",
        action="store_true",
        help="Whether to backend compile the transpiled graph",
    )
    parser.addoption(
        "--target",
        action="store",
        default="all",
        help="Target for the transpilation tests",
    )
    parser.addoption(
        "--no-logs",
        action="store_true",
        help="Do not include any optional ivy debugging logs.",
    )


def pytest_configure(config):
    getopt = config.getoption

    global BACKEND_COMPILE
    BACKEND_COMPILE = getopt("--backend-compile")

    global TARGET
    TARGET = getopt("--target")

    global NO_LOGS
    NO_LOGS = getopt("--no-logs")


def pytest_generate_tests(metafunc):
    if NO_LOGS: os.environ["DEBUG"] = "0"
    configs = list()
    if TARGET not in ["jax", "numpy", "tensorflow", "torch"]:
        for target in TARGET_FRAMEWORKS:
            configs.append((target, BACKEND_COMPILE))
    else:
        configs.append((TARGET, BACKEND_COMPILE))
    metafunc.parametrize("target_framework,backend_compile", configs)
