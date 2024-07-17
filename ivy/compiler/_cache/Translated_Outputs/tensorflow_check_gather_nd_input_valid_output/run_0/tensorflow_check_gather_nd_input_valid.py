from .tensorflow__helpers import tensorflow_get_item


def tensorflow_check_gather_nd_input_valid(params, indices, batch_dims):
    if batch_dims >= len(params.shape):
        raise Exception(
            f"batch_dims = {batch_dims} must be less than rank(`params`) = {len(params.shape)}."
        )
    if batch_dims >= len(indices.shape):
        raise Exception(
            f"batch_dims = {batch_dims}  must be less than rank(`indices`) = {len(indices.shape)}."
        )
    if tensorflow_get_item(
        params.shape, slice(0, batch_dims, None)
    ) != tensorflow_get_item(indices.shape, slice(0, batch_dims, None)):
        raise Exception(
            f"batch dimensions must match in `params` and `indices`; saw {tensorflow_get_item(params.shape, slice(0, batch_dims, None))} vs. {tensorflow_get_item(indices.shape, slice(0, batch_dims, None))}"
        )
    if indices.shape[-1] > len(
        tensorflow_get_item(params.shape, slice(batch_dims, None, None))
    ):
        raise Exception(
            f"index innermost dimension length must be <= rank(`params[batch_dims:]`); saw: {indices.shape[-1]} vs. {len(tensorflow_get_item(params.shape, slice(batch_dims, None, None)))} ."
        )
