from .tensorflow__helpers import tensorflow_exists_bknd


def tensorflow_check_exists(x, inverse=False, message=""):
    if inverse and tensorflow_exists_bknd(x):
        raise Exception("arg must be None" if message == "" else message)
    elif not inverse and not tensorflow_exists_bknd(x):
        raise Exception("arg must not be None" if message == "" else message)
