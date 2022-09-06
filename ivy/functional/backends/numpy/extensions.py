def is_native_sparse_array(x):
    """
    NumPy does not support sparse arrays.
    """
    return False


def init_data_sparse_array(indices, values, shape):
    return None


def init_native_components(x):
    """
    NumPy does not support sparse arrays.
    """
    return None, None, None
