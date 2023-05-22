import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def tanh(input):
    return ivy.tanh(input)


@to_ivy_arrays_and_back
def gelu(input, method="none"):
    if method == "none":
        return ivy.gelu(input, approximate=False)
    elif method == "tanh":
        return ivy.gelu(input, approximate=True)
    else:
        raise ivy.utils.exceptions.IvyException(
            "`method` argument must be either 'none' or 'tanh'."
        )


@to_ivy_arrays_and_back
def leaky_relu(input, alpha=0.2):
    return ivy.leaky_relu(input, alpha=alpha)


@to_ivy_arrays_and_back
def logsigmoid(input):
    return ivy.logsigmoid(input)


@to_ivy_arrays_and_back
def log_softmax(input, dim=None):
    if dim is None:
        dim = -1
    return ivy.log_softmax(input, axis=dim)


@to_ivy_arrays_and_back
def mish(input):
    return ivy.multiply(input, ivy.tanh(ivy.softplus(input)))


@to_ivy_arrays_and_back
def relu(input):
    return ivy.relu(input)


@to_ivy_arrays_and_back
def selu(input, inplace=False):
    ret_val = ivy.selu(input)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    prod = ivy.multiply(alpha, ivy.expm1(input))
    max_ = ivy.greater(input, 0)
    ret = ivy.multiply(scale, ivy.where(max_, input, prod))
    if ivy.is_complex_dtype(input):
        return ret
    if inplace:
        ivy.inplace_update(input, ret_val)
        return input
    return ret_val


@to_ivy_arrays_and_back
def sigmoid(input):
    return ivy.sigmoid(input)


@to_ivy_arrays_and_back
def silu(input, inplace=False):
    sigmoid = 1.0 / (1.0 + ivy.exp(-input))
    ret = ivy.multiply(input, sigmoid)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
def softmax(input, dim=None):
    if dim is None:
        dim = -1
    return ivy.softmax(input, axis=dim)


@to_ivy_arrays_and_back
def softplus(input):
    return ivy.softplus(input)


@to_ivy_arrays_and_back
def thresholded_relu(input, threshold, inplace=False):
    ret = ivy.where(ivy.greater(input, threshold), input, 0)
    if inplace:
        return ivy.inplace_update(input, ret)
    return ivy.thresholded_relu(input, threshold=threshold)


@to_ivy_arrays_and_back
def tanh(input):
    return ivy.tanh(input)


@to_ivy_arrays_and_back
def gelu(input, method="none"):
    if method == "none":
        return ivy.gelu(input, approximate=False)
    elif method == "tanh":
        return ivy.gelu(input, approximate=True)
    else:
        raise ivy.utils.exceptions.IvyException(
            "`method` argument must be either 'none' or 'tanh'."
        )


@to_ivy_arrays_and_back
def leaky_relu(input, a=0.2):
    return ivy.leaky_relu(input, alpha=a)


@to_ivy_arrays_and_back
def logsigmoid(input):
    return ivy.logsigmoid(input)


@to_ivy_arrays_and_back
def log_softmax(input, dim=None):
    if dim is None:
        dim = -1
    return ivy.log_softmax(input, axis=dim)


@to_ivy_arrays_and_back
def mish(input):
    return ivy.multiply(input, ivy.tanh(ivy.softplus(input)))


@to_ivy_arrays_and_back
def relu(input):
    return ivy.relu(input)


@to_ivy_arrays_and_back
def selu(input, inplace=False):
    ret_val = ivy.selu(input)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    prod = ivy.multiply(alpha, ivy.expm1(input))
    max_ = ivy.greater(input, 0)
    ret = ivy.multiply(scale, ivy.where(max_, input, prod))
    if ivy.is_complex_dtype(input):
        return ret
    if inplace:
        ivy.inplace_update(input, ret_val)
        return input
    return ret_val


@to_ivy_arrays_and_back
def sigmoid(input):
    return ivy.sigmoid(input)


@to_ivy_arrays_and_back
def silu(input, inplace=False):
    sigmoid_func = 1.0 / (1.0 + ivy.exp(-input))
    ret = ivy.multiply(input, sigmoid_func)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
def softmax(input, dim=None):
    if dim is None:
        dim = -1
    return ivy.softmax(input, axis=dim)


@to_ivy_arrays_and_back
def softplus(input):
    return ivy.softplus(input)


@to_ivy_arrays_and_back
def thresholded_relu(input, threshold, inplace=False):
    ret = ivy.where(ivy.greater(input, threshold), input, 0)
    if inplace:
        return ivy.inplace_update(input, ret)
    return ivy.thresholded_relu(input, threshold=threshold)


@to_ivy_arrays_and_back
def relu6(input, inplace=False):
    ret = ivy.relu6(input)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret
