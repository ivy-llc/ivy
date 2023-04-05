from importlib.util import find_spec
from ivy.utils.backend.handler import _backend_dict

# A list of available backends that can be used for testing.


def get_available_frameworks():
    available_frameworks = []
    for framework in _backend_dict:
        if find_spec(framework) is not None:
            available_frameworks.append(framework)
    return available_frameworks


def ground_truth():
    available_framework_lis = get_available_frameworks()
    g_truth = ""
    if "tensorflow" in available_framework_lis:
        g_truth = "tensorflow"
    elif "torch" in available_framework_lis:
        g_truth = "torch"
    elif "jax" in available_framework_lis:
        g_truth = "jax"
    elif "paddle" in available_framework_lis:
        g_truth = "paddle"
    else:
        g_truth = "numpy"
    return g_truth
