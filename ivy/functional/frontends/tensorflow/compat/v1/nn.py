# local
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


# should have float16 as well but sqrt doesn't support it
@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.9.0 and below": ("float32",)}, "tensorflow"
)
def fused_batch_norm(
    x,
    scale,
    offset,
    mean=None,
    variance=None,
    epsilon=1e-3,
    data_format='NHWC',
    is_training=True,
    name=None,
    exponential_avg_factor=1.0,
):
    dims = len(x.shape)
    if data_format[1] == "C":
        if dims == 4:
            x = ivy.permute_dims(x, axes=(0, 2, 3, 1))
        elif dims == 5:
            x = ivy.permute_dims(x, axes=(0, 2, 3, 4, 1))
        else:
            raise ivy.exceptions.IvyException(
                "input tensor must be of 4 or 5 dimensions, got {}".format(dims)
            )

    scale = scale.astype(ivy.float32)
    offset = offset.astype(ivy.float32)
    old_mean = mean.astype(ivy.float32)
    old_var = variance.astype(ivy.float32)
    x = x.astype(ivy.float32)

    if is_training:
        depth = x.shape[-1]
        rest_size = ivy.prod(x.shape) // depth
        x_rest_by_depth = ivy.reshape(x, [rest_size, depth])
        mean = ivy.mean(x_rest_by_depth, axis=0, keepdims=True)
        variance = ivy.var(x_rest_by_depth, axis=0, keepdims=True)
        y = ivy.reshape(
            scale * (x_rest_by_depth - mean) / ivy.sqrt(variance + epsilon) + offset,
            x.shape
        )
        variance = variance * rest_size / (rest_size - 1) if rest_size > 1 else variance
        mean = ivy.reshape(
            mean * exponential_avg_factor + old_mean * (1 - exponential_avg_factor),
            old_mean.shape
        )
        variance = ivy.reshape(
            variance * exponential_avg_factor + old_var * (1 - exponential_avg_factor),
            old_var.shape
        )
    else:
        y = scale * (x - old_mean) / ivy.sqrt(old_var + epsilon) + offset

    # permute dimensions back
    if data_format[1] == "C":
        if dims == 4:
            y = ivy.permute_dims(y, axes=(0, 3, 1, 2))
        elif dims == 5:
            y = ivy.permute_dims(y, axes=(0, 4, 1, 2, 3))

    if is_training:
        return y, mean, variance
    else:
        return y, old_mean, old_var
