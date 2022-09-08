def is_native_sparse_array(x):
    """Jax does not support sparse arrays."""
    return False


def init_data_sparse_array(indices, values, shape):
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    """Jax does not support sparse arrays."""
    return None, None, None
