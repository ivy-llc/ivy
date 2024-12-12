# TODO: Look for an elegant way to migrate these to the config files
import ivy

DATA_FORMAT = "PT"  # data format (eg: NCHW , NHWC)

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
CONV_ND = None  # conv dimension (eg: 1D, 2D, 3D)

ALL_IVY_DECORATORS = (
    ivy.func_wrapper.FN_DECORATORS
    + [
        "_asarray_to_native_arrays_and_back",
        "_asarray_infer_device",
        "_asarray_handle_nestable",
        "_asarray_inputs_to_native_shapes",
        "_asarray_infer_dtype",
        "_handle_nestable_dtype_info",
        "to_native_arrays_and_back",
    ]
    + [
        "with_unsupported_dtypes",
        "with_supported_dtypes",
        "with_unsupported_devices",
        "with_supported_devices",
        "with_unsupported_device_and_dtypes",
        "with_supported_device_and_dtypes",
    ]
)

ALL_IVY_FRONTEND_DECORATORS = (
    # torch frontend
    [
        "inputs_to_ivy_arrays",
        "numpy_to_torch_style_args",
        "outputs_to_frontend_arrays",
        "outputs_to_native_arrays",
        "to_ivy_arrays_and_back",
        "to_native_arrays_and_back",
        "to_ivy_shape",
        "handle_exceptions",
    ]
    + [
        "with_unsupported_dtypes",
        "with_supported_dtypes",
        "with_unsupported_devices",
        "with_supported_devices",
        "with_unsupported_device_and_dtypes",
        "with_supported_device_and_dtypes",
    ]
    # numpy frontend
    + [
        "handle_numpy_dtype",
    ]
)

IVY_DECORATORS_TO_TRANSLATE = [
    "handle_array_like_without_promotion",
    "infer_dtype",
    "_asarray_infer_dtype",
    "_asarray_to_native_arrays_and_back",
    "handle_partial_mixed_function",
]

BUILTIN_DECORATORS = [
    "property",
]

LIST_COMP_TRANSFORMATION_BLOCKLIST = ALL_IVY_DECORATORS + ALL_IVY_FRONTEND_DECORATORS

# set of conflicting method calls
CONFLICTING_METHODS = set()

# list of HF utility classes which are directly imported and not translated.
CLASSES_TO_IGNORE = (
    "BaseModelOutput",
    "ModelOutput",
    "ModuleUtilsMixin",
    "GenerationMixin",
    "PreTrainedModel",
    "PretrainedConfig",
    "transformers.modeling_outputs.BaseModelOutput",
    "transformers.modeling_outputs.ModelOutput",
    "transformers.utils.ModelOutput",
    "transformers.modeling_utils.ModuleUtilsMixin",
    "transformers.modeling_utils.GenerationMixin",
    "transformers.modeling_utils.PreTrainedModel",
    "transformers.configuration_utils.PretrainedConfig",
)
