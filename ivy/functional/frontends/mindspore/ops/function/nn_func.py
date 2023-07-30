"""Includes Mindspore Frontend functions listed in the TODO list
https://github.com/unifyai/ivy/issues/14951."""

# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@with_supported_dtypes(
    {
        "2.0.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        )
    },
    "mindspore",
)
@to_ivy_arrays_and_back
def dropout2d(input, p=0.5, training=True):
    return ivy.dropout2d(input, p, training=training, data_format="NCHW")


@with_supported_dtypes({"2.0.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def selu(input_x):
    return ivy.selu(input_x)


@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def softsign(x):
    return ivy.divide(x, ivy.add(1, ivy.abs(x)))

@with_supported_dtypes({"2.0.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def adaptive_avg_pool3d(input, output_size):
    return ivy.adaptive_avg_pool3d(input, output_size, data_format="NCDHW")