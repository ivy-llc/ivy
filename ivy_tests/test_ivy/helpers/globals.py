"""
A state holder for testing, this is only intended to hold and store testing data to be
used by the test helpers to prune unsupported data.

Should not be used inside any of the test functions.
"""


from dataclasses import dataclass

# needed for multiversion
available_frameworks = ["numpy", "jax", "tensorflow", "torch", "paddle", "mxnet"]
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

if "paddle" in available_frameworks:
    FWS_DICT["paddle"] = lambda x=None: _get_ivy_paddle(x)

if "mxnet" in available_frameworks:
    FWS_DICT["mxnet"] = lambda x=None: _get_ivy_mxnet(x)


# This is used to make sure the variable is not being overriden
_Notsetval = object()
CURRENT_GROUND_TRUTH_BACKEND: callable = _Notsetval
CURRENT_BACKEND: callable = _Notsetval
CURRENT_FRONTEND: callable = _Notsetval
CURRENT_RUNNING_TEST = _Notsetval
CURRENT_DEVICE = _Notsetval
CURRENT_DEVICE_STRIPPED = _Notsetval
CURRENT_FRONTEND_STR = None


@dataclass(frozen=True)  # ToDo use kw_only=True when version is updated
class TestData:
    test_fn: callable
    fn_tree: str
    fn_name: str
    supported_device_dtypes: dict = None
    is_method: bool = False


class InterruptedTest(BaseException):
    """Indicate that a test tried to write global attributes while a test is running."""

    def __init__(self, test_interruped):
        super.__init__(f"{test_interruped} was interruped during execution.")


def _get_ivy_numpy(version=None):
    """Import Numpy module from ivy."""
    try:
        import ivy.functional.backends.numpy
    except ImportError:
        return None
    return ivy.functional.backends.numpy


def _get_ivy_jax(version=None):
    """Import JAX module from ivy."""
    try:
        import ivy.functional.backends.jax
    except ImportError:
        return None
    return ivy.functional.backends.jax


def _get_ivy_tensorflow(version=None):
    """Import Tensorflow module from ivy."""
    try:
        import ivy.functional.backends.tensorflow
    except ImportError:
        return None
    return ivy.functional.backends.tensorflow


def _get_ivy_torch(version=None):
    """Import Torch module from ivy."""
    try:
        import ivy.functional.backends.torch
    except ImportError:
        return None
    return ivy.functional.backends.torch


def _get_ivy_paddle(version=None):
    """Import Paddle module from ivy."""
    try:
        import ivy.functional.backends.paddle
    except ImportError:
        return None
    return ivy.functional.backends.paddle


def _get_ivy_mxnet(version=None):
    """Import mxnet module from ivy."""
    try:
        import ivy.functional.backends.mxnet
    except ImportError:
        return None
    return ivy.functional.backends.mxnet


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
    CURRENT_FRONTEND_STR = framework
    CURRENT_FRONTEND = FWS_DICT[framework]


def _set_backend(framework: str):
    global CURRENT_BACKEND
    if CURRENT_BACKEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_BACKEND = FWS_DICT[framework]


def _set_ground_truth_backend(framework: str):
    global CURRENT_GROUND_TRUTH_BACKEND
    if CURRENT_GROUND_TRUTH_BACKEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_GROUND_TRUTH_BACKEND = FWS_DICT[framework]


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
