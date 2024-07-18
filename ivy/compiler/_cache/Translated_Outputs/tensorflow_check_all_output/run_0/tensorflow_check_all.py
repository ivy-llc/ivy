from .tensorflow__helpers import tensorflow_all


def tensorflow_check_all(results, message="one of the args is False", as_array=True):
    if as_array and not tensorflow_all(results) or not as_array and not all(results):
        raise Exception(message)
