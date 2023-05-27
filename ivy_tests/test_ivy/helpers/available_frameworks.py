from importlib.util import find_spec

# A list of available backends that can be used for testing.


# TODO update calls to function to be static
# shouldn't do computation more than once, do it and store in a constant
def available_frameworks():
    ret = []
    for backend in ["numpy", "jax", "tensorflow", "torch", "paddle"]:
        if find_spec(backend) is not None:
            ret.append(backend)
    return ret
