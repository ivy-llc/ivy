# local
import ivy
from ivy import with_supported_dtypes, to_ivy_arrays_and_back


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def instance_norm(x,
                  running_mean=None,
                  running_var=None,
                  weight=None,
                  bias=None,
                  use_input_stats=True,
                  momentum=0.9,
                  eps=1e-05,
                  data_format='NCS',
                  name=None):
    n_dims = len(x.shape)
    if data_format == "NCS":
        x = ivy.permute_dims(x, axes=(*range(2, n_dims), 0, 1))
    elif data_format == "NSC":
        x = ivy.permute_dims(x, axes=(*range(1, n_dims - 1), 0, n_dims - 1))
    else:
        raise ValueError(f"Invalid data_format: {data_format}.")

    N = x.shape[-2]
    C = x.shape[-1]
    S = x.shape[0:-2]
    x = x.reshape((1, *S, N * C))
    mean = ivy.tile(running_mean, N)
    variance = ivy.tile(running_var, N)
    scale = ivy.tile(weight, N)
    offset = ivy.tile(bias, N)
    x_normalized, running_mean, running_variance = ivy.batch_norm(
        x,
        mean,
        variance,
        scale=scale,
        offset=offset,
        training=use_input_stats,
        eps=eps,
        momentum=momentum,
    )
    x_normalized = x_normalized.reshape((*S, N, C))

    if data_format == "NCS":
        x_normalized = ivy.permute_dims(
            x_normalized, axes=(n_dims - 2, n_dims - 1, *range(0, n_dims - 2))
        )
    else:
        x_normalized = ivy.permute_dims(
            x_normalized, axes=(n_dims - 2, *range(0, n_dims - 2), n_dims - 1)
        )

    running_mean = running_mean.reshape((N, C)).mean(axis=0)
    running_variance = running_variance.reshape((N, C)).mean(axis=0)

    return x_normalized, running_mean, running_variance
