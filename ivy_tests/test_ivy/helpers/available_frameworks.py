from importlib.util import find_spec
import os

# A list of available backends that can be used for testing.


def _available_frameworks():
    ret = []
    for backend in ["numpy", "jax", "tensorflow", "torch", "paddle"]:
        if find_spec(backend) is not None:
            ret.append(backend)
        elif os.path.exists(f"/opt/fw/{backend}"):
            ret.append(backend)

    return ["numpy", "jax", "tensorflow", "torch"]


available_frameworks = _available_frameworks()
