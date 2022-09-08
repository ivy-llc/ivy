import logging


def is_native_sparse_array(x):
    """Numpy does not support sparse arrays."""
    return False


def native_sparse_array(data=None, *, indices=None, values=None, dense_shape=None):
    logging.warning("Numpy does not support sparse arrays, None is returned.")
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    """Numpy does not support sparse arrays."""
    return None, None, None
