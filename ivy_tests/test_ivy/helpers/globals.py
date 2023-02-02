"""
A state holder for testing, this is only intended to hold and store
testing data to be used by the test helpers to prune unsupported data.
Should not be used inside any of the test functions.
"""
import sys
from ... import config


from dataclasses import dataclass

# needed for multiversion
available_frameworks = ["numpy", "jax", "tensorflow", "torch"]
FWS_DICT = {
    "": lambda: None,
}

if "numpy" in available_frameworks:
    FWS_DICT["numpy"] = lambda x=None: _get_ivy_numpy(x)

if "jax" in available_frameworks:
    FWS_DICT["jax"] = lambda x=None: _get_ivy_jax(x)

if "tensorflow" in available_frameworks:
    FWS_DICT["tensorflow"] = lambda x=None: _get_ivy_tensorflow(x)
    FWS_DICT["tensorflow_graph"] = lambda: _get_ivy_tensorflow()

if "torch" in available_frameworks:
    FWS_DICT["torch"] = lambda x=None: _get_ivy_torch(x)


# This is used to make sure the variable is not being overriden
_Notsetval = object()
CURRENT_GROUND_TRUTH_BACKEND: callable = _Notsetval
CURRENT_BACKEND: callable = _Notsetval
CURRENT_FRONTEND: callable = _Notsetval
CURRENT_RUNNING_TEST = _Notsetval


@dataclass(frozen=True)  # ToDo use kw_only=True when version is updated
class TestData:
    test_fn: callable
    fn_tree: str
    fn_name: str
    supported_device_dtypes: dict = None


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


def _get_ivy_numpy(version=None):
    """Import Numpy module from ivy"""
    if version:
        config.reset_sys_modules_to_base()
        config.allow_global_framework_imports(fw=[version])

    try:
        import ivy.functional.backends.numpy
    except ImportError:
        return None
    return ivy.functional.backends.numpy


def _get_ivy_jax(version=None):
    """Import JAX module from ivy"""
    if version:
        las = [
            version.split("/")[0] + "/" + version.split("/")[1],
            version.split("/")[2] + "/" + version.split("/")[3],
        ]
        config.allow_global_framework_imports(fw=las)
        try:
            config.reset_sys_modules_to_base()
            import ivy.functional.backends.jax

            return ivy.functional.backends.jax
        except ImportError as e:
            raise e
    else:
        try:
            import ivy.functional.backends.jax
        except ImportError:
            return None
    return ivy.functional.backends.jax


def _get_ivy_tensorflow(version=None):
    """Import Tensorflow module from ivy"""
    if version:
        config.allow_global_framework_imports(fw=[version])
    try:
        import ivy.functional.backends.tensorflow
    except ImportError:
        return None
    return ivy.functional.backends.tensorflow


def _get_ivy_torch(version=None):
    """Import Torch module from ivy"""
    if version:
        config.allow_global_framework_imports(fw=[version])
    try:
        import ivy.functional.backends.torch
    except ImportError:
        return None
    return ivy.functional.backends.torch


# Setup


def setup_api_test(test_data: TestData, backend: str, ground_truth_backend: str):
    _set_test_data(test_data)
    _set_backend(backend)
    _set_ground_truth_backend(ground_truth_backend)


def teardown_api_test():
    _unset_test_data()
    _unset_backend()
    _unset_ground_truth_backend()


def setup_frontend_test(test_data: TestData, frontend: str, backend: str):
    _set_test_data(test_data)
    _set_frontend(frontend)
    _set_backend(backend)


def teardown_frontend_test():
    _unset_test_data()
    _unset_frontend()
    _unset_backend()


def _set_test_data(test_data: TestData):
    global CURRENT_RUNNING_TEST
    if CURRENT_RUNNING_TEST is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_RUNNING_TEST = test_data


def _set_frontend(framework: str):
    global CURRENT_FRONTEND
    if CURRENT_FRONTEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    if "/" in framework:
        CURRENT_FRONTEND = FWS_DICT[framework.split("/")[0]]
    else:
        CURRENT_FRONTEND = FWS_DICT[framework]


def _set_backend(framework: str):
    global CURRENT_BACKEND
    if CURRENT_BACKEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    if "/" in framework:
        pass
    CURRENT_BACKEND = FWS_DICT[framework]


def _set_ground_truth_backend(framework: str):
    global CURRENT_GROUND_TRUTH_BACKEND
    if CURRENT_GROUND_TRUTH_BACKEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_GROUND_TRUTH_BACKEND = FWS_DICT[framework]


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
