import ivy
import pytest

TARGET_FRAMEWORKS = ["numpy", "jax", "tensorflow", "torch"]
BACKEND_COMPILE = False


@pytest.fixture(autouse=True)
def run_around_tests():
    ivy.unset_backend()


def pytest_addoption(parser):
    parser.addoption(
        "--backend-compile",
        action="store_true",
        help="",
    )


def pytest_configure(config):
    getopt = config.getoption

    global BACKEND_COMPILE
    BACKEND_COMPILE = getopt("--backend-compile")


def pytest_generate_tests(metafunc):
    configs = list()
    for target in TARGET_FRAMEWORKS:
        configs.append((target, "transpile", BACKEND_COMPILE))
    configs.append(("torch", "trace", BACKEND_COMPILE))
    metafunc.parametrize("target_framework,mode,backend_compile", configs)
