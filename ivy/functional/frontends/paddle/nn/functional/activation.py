import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def _tanh(input_x):
    return ivy.tanh(input_x)


@to_ivy_arrays_and_back
def _gelu(input_x, method="none"):
    if method == "none":
        return ivy.gelu(input_x, approximate=False)
    elif method == "tanh":
        return ivy.gelu(input_x, approximate=True)
    else:
        raise ivy.utils.exceptions.IvyException(
            "`method` argument must be either 'none' or 'tanh'."
        )


@to_ivy_arrays_and_back
def _leaky_relu(input_x, alpha=0.2):
    return ivy.leaky_relu(input_x, alpha=alpha)


@to_ivy_arrays_and_back
def _logsigmoid(input_x):
    return ivy.logsigmoid(input_x)


@to_ivy_arrays_and_back
def _log_softmax(input_x, dim=None):
    if dim is None:
        dim = -1
    return ivy.log_softmax(input_x, axis=dim)


@to_ivy_arrays_and_back
def _mish(input_x):
    return ivy.multiply(input_x, ivy.tanh(ivy.softplus(input_x)))


@to_ivy_arrays_and_back
def _relu(input_x):
    return ivy.relu(input_x)


@to_ivy_arrays_and_back
def _selu(input_x, inplace=False):
    ret_val = ivy.selu(input_x)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    prod = ivy.multiply(alpha, ivy.expm1(input_x))
    max_ = ivy.greater(input_x, 0)
    ret = ivy.multiply(scale, ivy.where(max_, input_x, prod))
    if ivy.is_complex_dtype(input_x):
        return ret
    if inplace:
        ivy.inplace_update(input_x, ret_val)
        return input_x
    return ret_val


@to_ivy_arrays_and_back
def _sigmoid(input_x):
    return ivy.sigmoid(input_x)


@to_ivy_arrays_and_back
def _silu(input_x, inplace=False):
    sigmoid_func = 1.0 / (1.0 + ivy.exp(-input_x))
    ret = ivy.multiply(input_x, sigmoid_func)
    if inplace:
        ivy.inplace_update(input_x, ret)
        return input_x
    return ret


@to_ivy_arrays_and_back
def _softmax(input_x, dim=None):
    if dim is None:
        dim = -1
    return ivy.softmax(input_x, axis=dim)


@to_ivy_arrays_and_back
def _softplus(input_x):
    return ivy.softplus(input_x)


@to_ivy_arrays_and_back
def _thresholded_relu(input_x, threshold, inplace=False):
    ret = ivy.where(ivy.greater(input_x, threshold), input_x, 0)
    if inplace:
        return ivy.inplace_update(input_x, ret)
    return ivy.thresholded_relu(input_x, threshold=threshold)
