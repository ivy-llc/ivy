import logging


def is_native_sparse_array(x):
    """Jax does not support sparse arrays natively."""
    return False


def native_sparse_array(data=None, *, indices=None, values=None, dense_shape=None):
    logging.warning("Jax does not support sparse arrays natively, None is returned.")
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    logging.warning("Jax does not support sparse arrays natively, None is returned.")
    return None, None, None
