import functools
import re
import inspect
from enum import Enum
import ast
import copy
import os
import ivy

DATA_FORMAT = "PT"
CONV_FUNCS = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
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
KERAS_PADDING_FUNCS = [
    "KerasZeroPadding1D",
    "KerasZeroPadding2D",
    "KerasZeroPadding3D",
]
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
DROPOUT_FUNCS = [
    "Dropout",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
]
KERAS_DROPOUT_FUNCS = [
    "KerasDropout",
]

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


def handle_methods(fn):
    def extract_function_name(s):
        match = re.search(r"_(.+?)(?:_\d+)?$", s)
        if match:
            return match.group(1)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if ivy.is_array(args[0]):
            return fn(*args, **kwargs)
        else:
            fn_name = extract_function_name(fn.__name__)
            new_fn = getattr(args[0], fn_name)
            return new_fn(*args[1:], **kwargs)

    return wrapper


def handle_get_item(fn):
    @functools.wraps(fn)
    def wrapper(inp, query, **kwargs):
        try:
            res = inp.__getitem__(query)
        except IndexError:
            raise
        except Exception:
            res = fn(inp, query, **kwargs)
        return res

    return wrapper


def handle_set_item(fn):
    @functools.wraps(fn)
    def wrapper(inp, query, val, **kwargs):
        try:
            inp.__setitem__(query, val)
            res = inp
        except IndexError:
            raise
        except Exception:
            res = fn(inp, query, val, **kwargs)
        return res

    return wrapper


def store_config_info(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        # store trackable layer info
        if all(
            [
                hasattr(self, "_args"),
                hasattr(self, "_kwargs"),
                hasattr(self, "_self_tracked_trackables"),
            ]
        ):
            orig_trackables = copy.copy(self._self_tracked_trackables)
            self._args = (self,) + args
            self._kwargs = kwargs
            self._self_tracked_trackables = orig_trackables

    return wrapper


def retrieve_object(frame, name):
    if name is None:
        return name

    names = name.split(".")
    obj = frame.f_locals.get(names[0]) or frame.f_globals.get(names[0])
    if obj is None:
        return None

    for attr in names[1:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None

    return obj


class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.func_name = None

    def visit_Call(self, node):
        self.func_name = ast.unparse(node.func).strip()
        return super().generic_visit(node)


def get_next_func(obj):

    # Traverse down the stack to reach the first occurrence of the function named "call"
    stack = inspect.stack()
    for frame_info in stack:
        if frame_info == obj._previous_frame_info:
            calling_frame = frame_info.frame
            break
    else:
        return None

    # Check if the filename contains "Sequential"
    if "Sequential" in frame_info.filename:
        try:
            # find the next call in the sequence
            self_seq = calling_frame.f_locals["self"]
            idx = calling_frame.f_locals["i"]
            next_func = self_seq[idx + 1]
            return next_func
        except IndexError:
            # If it fails, traverse further down the stack frame
            # to find the next occurrence of "call"
            for frame_info in stack[stack.index(frame_info) + 1 :]:
                if frame_info == self_seq._previous_frame_info:
                    calling_frame = frame_info.frame
                    break
            else:
                return None
    # Retrieve the source code for the immediate next statement
    # following the one we are executing
    lines, start_line_no = inspect.getsourcelines(calling_frame)
    current_line_no = calling_frame.f_lineno
    relative_line_no = current_line_no - start_line_no
    # Parse the next statement to get the next call
    try:
        next_line = lines[relative_line_no + 1].strip()
        tree = ast.parse(next_line)
        visitor = CallVisitor()
        visitor.visit(tree)
        next_call_str = visitor.func_name
    except Exception:
        next_call_str = ""

    # Retrieve the original object from the calling frame
    next_func = retrieve_object(calling_frame, next_call_str)

    return next_func


class TransposeType(Enum):
    """Possible transpose types."""

    NO_TRANSPOSE = "no_transpose"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    CONV3D = "conv3d"


def apply_transpose(input, transpose, pt_to_tf=True):
    if transpose is TransposeType.NO_TRANSPOSE:
        return input

    if transpose is TransposeType.CONV1D:
        # Conv1D input:
        #    PT: (num_out_channel, num_in_channel, kernel)
        # -> TF: (kernel, num_in_channel, num_out_channel)
        axes = (0, 2, 1) if pt_to_tf else (0, 2, 1)
    elif transpose is TransposeType.CONV2D:
        # Conv2D input:
        #    PT: (num_out_channel, num_in_channel, kernel[0], kernel[1])
        # -> TF: (kernel[0], kernel[1], num_in_channel, num_out_channel)
        axes = (0, 2, 3, 1) if pt_to_tf else (0, 3, 1, 2)
    elif transpose is TransposeType.CONV3D:
        # Conv3D input:
        #    PT: (num_out_channel, num_in_channel, kernel[0], kernel[1], kernel[2])
        # -> TF: (kernel[0], kernel[1], kernel[2], num_in_channel, num_out_channel)
        axes = (0, 2, 3, 4, 1) if pt_to_tf else (0, 4, 1, 2, 3)

    input = ivy.permute_dims(input, axes=axes).data
    return input


def handle_transpose_in_input_and_output(fn):

    original_signature = inspect.signature(fn)

    @functools.wraps(fn)
    def transpose_wrapper(self, *args, **kwargs):
        global DATA_FORMAT
        # isolates the actual `**kwargs` for the decorated function
        kwargs_call = {
            key: val
            for key, val in kwargs.items()
            if key not in dict(original_signature.parameters)
        }
        fn_args_and_kwargs = {
            key: val for key, val in kwargs.items() if key not in kwargs_call
        }
        # move any arg into kwargs, if they exist
        fn_args_and_kwargs.update(dict(zip(fn.__code__.co_varnames[1:], args)))

        conv_block_start = lambda f: any(
            substr in f.__qualname__
            for substr in (
                CONV_FUNCS
                + NORM_FUNCS
                + POOL_FUNCS
                + KERAS_CONV_FUNCS
                + KERAS_NORM_FUNCS
                + KERAS_POOL_FUNCS
            )
        )
        next_call_in_seq = get_next_func(self)
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
                transpose = TransposeType.CONV3D
            elif len(input.shape) > 3:
                transpose = TransposeType.CONV2D
            elif len(input.shape) > 2:
                transpose = TransposeType.CONV1D
            else:
                transpose = TransposeType.NO_TRANSPOSE
            fn_args_and_kwargs["input"] = apply_transpose(
                input, transpose=transpose, pt_to_tf=True
            )

            DATA_FORMAT = "TF"
            os.environ["DATA_FORMAT"] = "channels_last"

        res = fn(self, **fn_args_and_kwargs)

        if (DATA_FORMAT == "TF" and conv_block_continued) or DATA_FORMAT == "PT":
            return res

        if len(res.shape) > 4:
            transpose = TransposeType.CONV3D
        elif len(res.shape) > 3:
            transpose = TransposeType.CONV2D
        elif len(res.shape) > 2:
            transpose = TransposeType.CONV1D
        else:
            transpose = TransposeType.NO_TRANSPOSE
        res = apply_transpose(res, transpose=transpose, pt_to_tf=False)

        DATA_FORMAT = "PT"
        os.environ["DATA_FORMAT"] = "channels_first"

        return res

    handle_transpose_in_input_and_output.__signature__ = original_signature
    return transpose_wrapper
