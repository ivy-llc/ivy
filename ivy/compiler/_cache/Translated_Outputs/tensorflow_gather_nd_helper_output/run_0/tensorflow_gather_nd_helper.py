import tensorflow


def tensorflow_gather_nd_helper(params, indices):
    indices_shape = tensorflow.shape(indices)
    params_shape = tensorflow.shape(params)
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        tensorflow.math.reduce_prod(params_shape[i + 1 :])
        for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = tensorflow.convert_to_tensor(
        result_dim_sizes_list, dtype=indices.dtype
    )
    implicit_indices_factor = result_dim_sizes[num_index_dims - 1]
    flat_params = tensorflow.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = tensorflow.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = tensorflow.reshape(
        tensorflow.reduce_sum(indices * indices_scales, -1, keepdims=True), (-1, 1)
    )
    indices_for_flat_tiled = tensorflow.repeat(
        indices_for_flat_tiled, implicit_indices_factor, axis=1
    )
    implicit_indices = tensorflow.repeat(
        tensorflow.expand_dims(tensorflow.range(implicit_indices_factor), 0),
        indices_for_flat_tiled.shape[0],
        axis=0,
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = tensorflow.reshape(indices_for_flat, (-1,))
    flat_gather = tensorflow.gather(flat_params, flat_indices_for_flat)
    res = tensorflow.reshape(
        flat_gather,
        tensorflow.concat([indices_shape[:-1], params_shape[num_index_dims:]], 0),
    )
    return res
