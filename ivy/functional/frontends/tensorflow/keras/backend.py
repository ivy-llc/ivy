import functools
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend
from ivy.functional.frontends.tensorflow.func_wrapper import (
    _ivy_array_to_tensorflow,
    _to_ivy_array,
    to_ivy_arrays_and_back,
)


def bias_add(x, bias, data_format=None):
    if data_format is None:
        data_format = "channels_last"
    bias_shape = bias.shape
    if len(bias_shape) == 1:
        if data_format == "channels_first":
            return tf_frontend.nn.bias_add(x, bias, data_format="NC...")
        return tf_frontend.nn.bias_add(x, bias, data_format="N...C")
    if x.ndim in (3, 4, 5):
        if data_format == "channels_first":
            bias_reshape_axis = (1, bias_shape[-1]) + bias_shape[:-1]
            return x + tf_frontend.reshape(bias, bias_reshape_axis)
        return x + tf_frontend.reshape(bias, (1,) + bias_shape)
    return tf_frontend.nn.bias_add(x, bias)


@to_ivy_arrays_and_back
def depthwise_conv2d(
    x,
    depthwise_kernel,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
):
    data_format = "channels_last" if data_format is None else data_format
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: " + str(data_format))

    tf_data_format = "NHWC"
    permuted_x = False
    if data_format == "channels_first":
        if ivy.dev(x) == "cpu":
            x = tf_frontend.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
            permuted_x = True
        else:
            tf_data_format = "NCHW"

    padding = padding.upper()
    if padding not in {"VALID", "SAME"}:
        raise ValueError("Unknown padding: " + str(padding))

    if tf_data_format == "NHWC":
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    x = tf_frontend.nn.depthwise_conv2d(
        x,
        depthwise_kernel,
        strides=strides,
        padding=padding,
        dilations=dilation_rate,
        data_format=tf_data_format,
    )

    if permuted_x:
        x = tf_frontend.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


@to_ivy_arrays_and_back
def dot(x, y):
    return ivy.dot(x, y)


def mean(x, axis=None, keepdims=False):
    return tf_frontend.reduce_mean(x, axis, keepdims)


@to_ivy_arrays_and_back
def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
    @functools.wraps(step_function)
    def _new_step_function(*args, **kwargs):
        frontend_args = ivy.nested_map(
            _ivy_array_to_tensorflow, args, include_derived=True, shallow=False
        )
        frontend_kwargs = ivy.nested_map(
            _ivy_array_to_tensorflow, kwargs, include_derived=True, shallow=False
        )
        ret = step_function(*frontend_args, **frontend_kwargs)
        return ivy.nested_map(_to_ivy_array, ret, include_derived=True)

    return ivy.rnn(
        _new_step_function,
        inputs,
        initial_states,
        go_backwards=go_backwards,
        mask=mask,
        constants=constants,
        unroll=unroll,
        input_length=input_length,
        time_major=time_major,
        zero_output_for_mask=zero_output_for_mask,
        return_all_outputs=return_all_outputs,
    )


def sum(x, axis=None, keepdims=False):
    return tf_frontend.reduce_sum(x, axis, keepdims)
