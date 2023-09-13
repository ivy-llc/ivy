from importlib.util import find_spec

# A list of available backends that can be used for testing.


def _available_frameworks():
    ret = []
    for backend in ["numpy", "jax", "tensorflow", "torch", "paddle"]:
        if find_spec(backend) is not None:
            ret.append(backend)
    return ret


available_frameworks = _available_frameworks()
