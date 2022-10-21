from dataclasses import dataclass

FWS_DICT = {
    "numpy": lambda: _get_ivy_numpy(),
    "jax": lambda: _get_ivy_jax(),
    "tensorflow": lambda: _get_ivy_tensorflow(),
    "tensorflow_graph": lambda: _get_ivy_tensorflow(),
    "torch": lambda: _get_ivy_torch(),
    "": lambda: None,
}
# This is used to make sure the variable is not being overriden
_Notsetval = object()
CURRENT_BACKEND = _Notsetval
CURRENT_FRONTEND = _Notsetval
CURRENT_RUNNING_TEST = _Notsetval


@dataclass(frozen=True)  # ToDo use kw_only=True when version is updated
class TestData:
    test_fn: callable
    fn_tree: str
    unsupported_dtypes: dict = None


class InterruptedTest(BaseException):
    """
    Used to indicate that a test tried to write global attributes
    while a test is running.
    """

    def __init__(self, test_interruped):
        super.__init__(f"{test_interruped} was interruped during execution.")


def _get_ivy_numpy():
    """Import Numpy module from ivy"""
    try:
        import ivy.functional.backends.numpy
    except ImportError:
        return None
    return ivy.functional.backends.numpy


def _get_ivy_jax():
    """Import JAX module from ivy"""
    try:
        import ivy.functional.backends.jax
    except ImportError:
        return None
    return ivy.functional.backends.jax


def _get_ivy_tensorflow():
    """Import Tensorflow module from ivy"""
    try:
        import ivy.functional.backends.tensorflow
    except ImportError:
        return None
    return ivy.functional.backends.tensorflow


def _get_ivy_torch():
    """Import Torch module from ivy"""
    try:
        import ivy.functional.backends.torch
    except ImportError:
        return None
    return ivy.functional.backends.torch


# Setup


def set_test_data(test_data: TestData):
    global CURRENT_RUNNING_TEST
    if CURRENT_RUNNING_TEST is _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_RUNNING_TEST = test_data


def set_frontend(framework: str):
    global CURRENT_FRONTEND
    if CURRENT_FRONTEND is _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_FRONTEND = FWS_DICT[framework]


def set_backend(framework: str):
    global CURRENT_BACKEND
    if CURRENT_BACKEND is _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_BACKEND = FWS_DICT[framework]


# Teardown


def unset_test_data():
    global CURRENT_RUNNING_TEST
    CURRENT_RUNNING_TEST = _Notsetval


def unset_frontend():
    global CURRENT_FRONTEND
    CURRENT_FRONTEND = _Notsetval


def unset_backend():
    global CURRENT_BACKEND
    CURRENT_BACKEND = _Notsetval
