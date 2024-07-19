from .tensorflow__helpers import tensorflow_any


def tensorflow_check_any(results, message="all of the args are False", as_array=True):
    if as_array and not tensorflow_any(results) or not as_array and not any(results):
        raise Exception(message)
