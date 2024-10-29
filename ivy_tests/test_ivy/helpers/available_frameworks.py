from importlib.util import find_spec
import os

# A list of available backends that can be used for testing.


def _available_frameworks(path="/opt/fw/"):
    ret = []
    for backend in ["numpy", "jax", "tensorflow", "torch"]:
        if find_spec(backend) is not None:
            ret.append(backend)
        elif os.path.exists(f"{path}{backend}"):
            ret.append(backend)

    return ret


available_frameworks = _available_frameworks()
