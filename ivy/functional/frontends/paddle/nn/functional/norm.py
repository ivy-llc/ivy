# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None):
    return ivy.layer_norm(x, normalized_shape, weight, bias, epsilon)


# instance_norm
@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def instance_norm(
    x,
    running_mean=None,
    running_var=None,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.9,
    epsilon=1e-05,
    data_format="NCHW",
    name=None,
):
    r"""
    Compute the 2-D inverse discrete Fourier Transform.

    Parameters
    ----------
    x
        Input array of float data type. It's data type should be float32, float64.

    running_mean

        running mean. Default None.

    running_var
        running variance. Default None.

    weight
        The input array weight of instance_norm. Default: None.

    bias
        The input array bias of instance_norm. Default: None.

    use_input_stats
        Default True
    momentum
        The optional value used for the moving_mean and
        moving_var computation. Default: 0.9.
    eps
        An optional value added to the denominator
        for numerical stability. Default is 1e-5.
    data_format
        Specify the optional input data format, may be “NC”, “NCL”, “NCHW”
        or “NCDHW”. Defalut “NCHW”.
    name
        The optional name of the layer norm, layer_norm(). It's generally used
        as the prefix identification of output and weight in network layer

    Returns
    -------
        An array which represents the instance after applying instance normalization.
    """
    return ivy.instance_norm(
        x,
        running_mean,
        running_var,
        weight,
        bias,
        use_input_stats,
        momentum,
        epsilon,
        data_format,
    )
