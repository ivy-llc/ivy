from .tensorflow__helpers import tensorflow_batch_norm_1


def tensorflow_batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    normalized, mean, var = tensorflow_batch_norm_1(
        input,
        running_mean,
        running_var,
        offset=bias,
        scale=weight,
        training=training,
        eps=eps,
        momentum=momentum,
        data_format="NSC",
    )
    return normalized, mean, var
