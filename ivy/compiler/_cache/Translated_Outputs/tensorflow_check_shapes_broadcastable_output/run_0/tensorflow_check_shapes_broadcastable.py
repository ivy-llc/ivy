from .tensorflow__helpers import tensorflow_check_one_way_broadcastable


def tensorflow_check_shapes_broadcastable(var, data):
    if not tensorflow_check_one_way_broadcastable(var, data):
        raise Exception(f"Could not broadcast shape {data} to shape {var}.")
