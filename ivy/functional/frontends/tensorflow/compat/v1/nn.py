# local
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.9.0 and below": ("float32", "float16", "bfloat16")}, "tensorflow"
)
def fused_batch_norm(
    x,
    scale,
    offset,
    mean=None,
    variance=None,
    epsilon=0.001,
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
    if is_training:
        dims_to_avg = tuple(range(0, len(x.shape))[:-1])
        x_mean = ivy.mean(x, axis=dims_to_avg)
        x_var = ivy.var(x, axis=dims_to_avg)
        x_norm = scale * (x - x_mean) / ivy.sqrt(x_var + epsilon) + offset
    else:
        # dtype = ivy.promote_types(x, mean)
        x_norm = scale * (x - mean) / ivy.sqrt(variance + epsilon) + offset
    if data_format[1] == "C":
        if dims == 4:
            x_norm = ivy.permute_dims(x_norm, axes=(0, 3, 1, 2))
        elif dims == 5:
            x_norm = ivy.permute_dims(x_norm, axes=(0, 4, 1, 2, 3))
    if is_training:
        if exponential_avg_factor == 1.0:
            moving_mean = x_mean
            moving_var = x_var
        else:
            moving_mean = x_mean * exponential_avg_factor + mean * (
                        1 - exponential_avg_factor)
            moving_var = x_var * exponential_avg_factor + variance * (
                        1 - exponential_avg_factor)
        return x_norm, moving_mean, moving_var
    else:
        return x_norm, mean, variance
    