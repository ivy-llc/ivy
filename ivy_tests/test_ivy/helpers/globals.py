"""
A state holder for testing, this is only intended to hold and store
testing data to be used by the test helpers to prune unsupported data.
Should not be used inside any of the test functions.
"""
import importlib
import sys
from typing import List


from dataclasses import dataclass


# This is used to make sure the variable is not being overriden
_Notsetval = object()
CURRENT_GROUND_TRUTH_BACKEND: callable = _Notsetval
CURRENT_BACKEND: callable = _Notsetval
CURRENT_FRONTEND: callable = _Notsetval
CURRENT_RUNNING_TEST = _Notsetval
CURRENT_DEVICE = _Notsetval
CURRENT_DEVICE_STRIPPED = _Notsetval
CURRENT_FRONTEND_STR = None
_backends_to_test = []
_backends_to_test_lock = False


@dataclass(frozen=True)  # ToDo use kw_only=True when version is updated
class TestData:
    test_fn: callable
    fn_tree: str
    fn_name: str
    supported_device_dtypes: dict = None
    is_method: bool = False


def remove_all_current_framework(framework):
    temp = sys.modules
    hold = {}
    unhold = {}
    for key, item in sys.modules.items():
        if getattr(item, "__file__", None):
            if "/opt/miniconda/fw/" + framework in getattr(
                item, "__file__", "willywonka"
            ):
                hold[key] = item
            else:
                unhold[key] = item
        else:
            unhold[key] = item
    sys.modules.clear()
    first_diff = {k: hold[k] for k in set(hold) - set(temp)}
    second_diff = {k: unhold[k] for k in set(unhold) - set(first_diff)}
    if second_diff:
        unhold.update(second_diff)
    sys.modules.update(unhold)
    if "/opt/miniconda/fw/" + framework in sys.path:
        sys.path.remove("/opt/miniconda/fw/" + framework)
    return (hold, framework)


class InterruptedTest(BaseException):
    """
    Used to indicate that a test tried to write global attributes
    while a test is running.
    """

    def __init__(self, test_interruped):
        super.__init__(f"{test_interruped} was interruped during execution.")


# Helpers #####

_imported_backends = {}


def _import_backend(backend: str):
    if backend in _imported_backends:
        return _imported_backends[backend]
    if _backends_to_test_lock:
        raise RuntimeError("Trying to import a backend when it's locked.")
    try:
        imported_backend = importlib.import_module(f"ivy.functional.backends.{backend}")
    except ImportError as e:
        raise ImportError(
            f"Trying to run tests using {backend} backend but {backend} backend "
            "has failed to be imported."
        ) from e
    if backend == "jax":
        from jax.config import config

        config.update("jax_enable_x64", True)
    _imported_backends[backend] = imported_backend
    return imported_backend


# Setup


def setup_api_test(
    backend: str,
    ground_truth_backend: str,
    device: str,
    test_data: TestData = None,
):
    if test_data is not None:
        _set_test_data(test_data)
    _set_backend(backend)
    _set_device(device)
    _set_ground_truth_backend(ground_truth_backend)


def teardown_api_test():
    _unset_test_data()
    _unset_backend()
    _unset_device()
    _unset_ground_truth_backend()


def setup_frontend_test(test_data: TestData, frontend: str, backend: str, device: str):
    _set_test_data(test_data)
    _set_frontend(frontend)
    _set_backend(backend)
    _set_device(device)


def teardown_frontend_test():
    _unset_test_data()
    _unset_frontend()
    _unset_backend()
    _unset_device()


def _set_test_data(test_data: TestData):
    global CURRENT_RUNNING_TEST
    if CURRENT_RUNNING_TEST is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_RUNNING_TEST = test_data


def _set_frontend(framework: str):
    global CURRENT_FRONTEND
    global CURRENT_FRONTEND_STR
    if CURRENT_FRONTEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    if isinstance(framework, list):
        CURRENT_FRONTEND = _import_backend(framework[0].split("/")[0])
        CURRENT_FRONTEND_STR = framework
    else:
        CURRENT_FRONTEND = _import_backend(framework)


def _set_backend(framework: str):
    global CURRENT_BACKEND
    if CURRENT_BACKEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    if "/" in framework:
        pass
    CURRENT_BACKEND = _import_backend(framework)


def _set_ground_truth_backend(framework: str):
    global CURRENT_GROUND_TRUTH_BACKEND
    if CURRENT_GROUND_TRUTH_BACKEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    if isinstance(framework, list):
        CURRENT_GROUND_TRUTH_BACKEND = framework
    else:
        CURRENT_GROUND_TRUTH_BACKEND = _import_backend(framework)


def _set_device(device: str):
    global CURRENT_DEVICE, CURRENT_DEVICE_STRIPPED
    if CURRENT_DEVICE is not _Notsetval or CURRENT_DEVICE_STRIPPED is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_DEVICE = device
    CURRENT_DEVICE_STRIPPED = device.partition(":")[0]


# Teardown


def _unset_test_data():
    global CURRENT_RUNNING_TEST
    CURRENT_RUNNING_TEST = _Notsetval


def _unset_frontend():
    global CURRENT_FRONTEND
    CURRENT_FRONTEND = _Notsetval


def _unset_backend():
    global CURRENT_BACKEND
    CURRENT_BACKEND = _Notsetval


def _unset_ground_truth_backend():
    global CURRENT_GROUND_TRUTH_BACKEND
    CURRENT_GROUND_TRUTH_BACKEND = _Notsetval


def _unset_device():
    global CURRENT_DEVICE, CURRENT_DEVICE_STRIPPED
    CURRENT_DEVICE = _Notsetval
    CURRENT_DEVICE_STRIPPED = _Notsetval


def lock_backends_to_test():
    global _backends_to_test_lock
    _backends_to_test_lock = True


def add_backend_to_test(backend: str):
    global _backends_to_test_lock
    if _backends_to_test_lock:
        raise RuntimeError("Modifying backends is locked.")
    _import_backend(backend)
    _backends_to_test.append(backend)


def get_backends_to_test() -> List[str]:
    return _backends_to_test
