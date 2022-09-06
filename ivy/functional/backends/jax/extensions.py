def is_native_sparse_array(x):
    """
    Jax does not support sparse arrays.
    """
    return False


def init_data_sparse_array(indices, values, shape):
    return None


def init_native_components(x):
    """
    Jax does not support sparse arrays.
    """
    return None, None, None
