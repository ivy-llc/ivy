import inspect
import functools
import os

from .tensorflow__helpers import tensorflow_apply_transpose
from .tensorflow__helpers import tensorflow_get_next_func
from .tensorflow__helpers import tensorflow_set_item

CONV_FUNCS = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]
NORM_FUNCS = [
    "_BatchNorm",
    "_InstanceNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "SyncBatchNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LocalResponseNorm",
]
POOL_FUNCS = [
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "FractionalMaxPool2d",
    "LPPool1d",
    "LPPool2d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
]
KERAS_CONV_FUNCS = [
    "KerasConv1D",
    "KerasConv2D",
    "KerasConv3D",
    "KerasDepthwiseConv2D",
    "KerasConv1DTranspose",
    "KerasConv2DTranspose",
    "KerasConv3DTranspose",
]
KERAS_NORM_FUNCS = [
    "KerasBatchNorm1D",
    "KerasBatchNorm2D",
    "KerasBatchNorm3D",
    "KerasLayerNormalization",
    "KerasGroupNormalization",
    "KerasUnitNorm1D",
    "KerasUnitNorm2D",
    "KerasUnitNorm3D",
]
KERAS_POOL_FUNCS = [
    "KerasAveragePooling1D",
    "KerasAveragePooling2D",
    "KerasAveragePooling3D",
    "KerasMaxPool1D",
    "KerasMaxPool2D",
    "KerasMaxPool3D",
]
PADDING_FUNCS = [
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad2d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
]
KERAS_PADDING_FUNCS = ["KerasZeroPadding1D", "KerasZeroPadding2D", "KerasZeroPadding3D"]
ACTIVATION_FUNCS = [
    "ELU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "LeakyReLU",
    "PReLU",
    "ReLU",
    "ReLU6",
    "RReLU",
    "SELU",
    "CELU",
    "GELU",
    "Sigmoid",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
    "AdaptiveLogSoftmaxWithLoss",
]
KERAS_ACTIVATION_FUNCS = [
    "KerasReLU",
    "KerasPReLU",
    "KerasLeakyReLU",
    "KerasThresholdedReLU",
    "KerasELU",
    "KerasSoftmax",
]
DROPOUT_FUNCS = [
    "Dropout",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
]
KERAS_DROPOUT_FUNCS = ["KerasDropout"]
CONV_BLOCK_FNS = [
    *CONV_FUNCS,
    *KERAS_CONV_FUNCS,
    *POOL_FUNCS,
    *KERAS_POOL_FUNCS,
    *PADDING_FUNCS,
    *KERAS_PADDING_FUNCS,
    *ACTIVATION_FUNCS,
    *KERAS_ACTIVATION_FUNCS,
    *NORM_FUNCS,
    *KERAS_NORM_FUNCS,
    *DROPOUT_FUNCS,
    *KERAS_DROPOUT_FUNCS,
]
DATA_FORMAT = "PT"


def tensorflow_handle_transpose_in_input_and_output(fn):
    from .tensorflow_TransposeType import tensorflow_TransposeType

    original_signature = inspect.signature(fn)

    @functools.wraps(fn)
    def transpose_wrapper(self, *args, **kwargs):
        global DATA_FORMAT
        kwargs_call = {
            key: val
            for key, val in kwargs.items()
            if key not in dict(original_signature.parameters)
        }
        fn_args_and_kwargs = {
            key: val for key, val in kwargs.items() if key not in kwargs_call
        }
        fn_args_and_kwargs.update(dict(zip(fn.__code__.co_varnames[1:], args)))
        conv_block_start = lambda f: any(
            substr in f.__qualname__
            for substr in CONV_FUNCS
            + NORM_FUNCS
            + POOL_FUNCS
            + KERAS_CONV_FUNCS
            + KERAS_NORM_FUNCS
            + KERAS_POOL_FUNCS
        )
        next_call_in_seq = tensorflow_get_next_func(self)
        name_of_next_call = (
            next_call_in_seq.__class__.__name__
            if hasattr(next_call_in_seq, "__class__")
            else ""
        )
        conv_block_continued = next_call_in_seq and any(
            substr in name_of_next_call for substr in CONV_BLOCK_FNS
        )
        if DATA_FORMAT == "PT" and conv_block_start(self.__class__):
            input = fn_args_and_kwargs["input"]
            if len(input.shape) > 4:
                transpose = tensorflow_TransposeType.CONV3D
            elif len(input.shape) > 3:
                transpose = tensorflow_TransposeType.CONV2D
            elif len(input.shape) > 2:
                transpose = tensorflow_TransposeType.CONV1D
            else:
                transpose = tensorflow_TransposeType.NO_TRANSPOSE
            fn_args_and_kwargs = tensorflow_set_item(
                fn_args_and_kwargs,
                "input",
                tensorflow_apply_transpose(input, transpose=transpose, pt_to_tf=True),
            )
            DATA_FORMAT = "TF"
            os.environ = tensorflow_set_item(os.environ, "DATA_FORMAT", "channels_last")
        res = fn(self, **fn_args_and_kwargs)
        if DATA_FORMAT == "TF" and conv_block_continued or DATA_FORMAT == "PT":
            return res
        if len(res.shape) > 4:
            transpose = tensorflow_TransposeType.CONV3D
        elif len(res.shape) > 3:
            transpose = tensorflow_TransposeType.CONV2D
        elif len(res.shape) > 2:
            transpose = tensorflow_TransposeType.CONV1D
        else:
            transpose = tensorflow_TransposeType.NO_TRANSPOSE
        res = tensorflow_apply_transpose(res, transpose=transpose, pt_to_tf=False)
        DATA_FORMAT = "PT"
        os.environ = tensorflow_set_item(os.environ, "DATA_FORMAT", "channels_first")
        return res

    tensorflow_handle_transpose_in_input_and_output.__signature__ = original_signature
    return transpose_wrapper
