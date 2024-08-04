from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__stateful import store_frame_info
import tensorflow as tf
import keras
import collections
from itertools import repeat
from numbers import Number
import os
from packaging.version import parse as parse_package


def parse(x):
    n = 2
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


def _reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))


def _handle_padding_shape(padding, n, mode):
    padding = tuple(
        [
            (padding[i * 2], padding[i * 2 + 1])
            for i in range(int(len(padding) / 2) - 1, -1, -1)
        ]
    )
    if mode == "circular":
        padding = padding + ((0, 0),) * (n - len(padding))
    else:
        padding = ((0, 0),) * (n - len(padding)) + padding
    if mode == "circular":
        padding = tuple(list(padding)[::-1])
    return padding


def _to_tf_padding(pad_width, ndim):
    if isinstance(pad_width, Number):
        pad_width = [[pad_width] * 2] * ndim
    elif len(pad_width) == 2 and isinstance(pad_width[0], Number):
        pad_width = [pad_width] * ndim
    elif (
        isinstance(pad_width, (list, tuple))
        and isinstance(pad_width[0], (list, tuple))
        and len(pad_width) < ndim
    ):
        pad_width = pad_width * ndim
    return pad_width


@tensorflow_handle_array_like_without_promotion
def _pad(
    input,
    pad_width,
    /,
    *,
    mode="constant",
    stat_length=1,
    constant_values=0,
    end_values=0,
    reflect_type="even",
    **kwargs,
):
    pad_width = _to_tf_padding(pad_width, len(input.shape))
    if not isinstance(constant_values, (tf.Variable, tf.Tensor)):
        constant_values = tf.constant(constant_values)
    if constant_values.dtype != input.dtype:
        constant_values = tf.cast(constant_values, input.dtype)
    return tf.pad(input, pad_width, mode=mode, constant_values=constant_values)


def torch_pad(input, pad, mode="constant", value=0):
    # deal with any negative pad values
    if any([pad_value < 0 for pad_value in pad]):
        pad = list(pad)
        slices = []
        for n in reversed(range(len(pad) // 2)):
            i = n * 2
            j = i + 1
            start = None
            stop = None
            if pad[i] < 0:
                start = -pad[i]
                pad[i] = 0
            if pad[j] < 0:
                stop = pad[j]
                pad[j] = 0
            slices.append(slice(start, stop))
        ndim = len(input.shape)
        while len(slices) < ndim:
            slices.insert(0, slice(None))
        input = input[tuple(slices)]

    value = 0 if value is None else value
    mode_dict = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }
    if mode not in mode_dict:
        raise ValueError(f"Unsupported padding mode: {mode}")
    pad = _handle_padding_shape(pad, len(input.shape), mode)
    order = 0, 2, 3, 1
    pad = tuple(pad[i] for i in order)
    return _pad(input, pad, mode=mode_dict[mode], constant_values=value)


def resolve_convolution(*args, **kwargs):
    depthwise_multiplier = kwargs["groups"] // kwargs["filters"]
    if depthwise_multiplier < 1:
        return KerasConv2D(*args, **kwargs)
    else:
        return KerasDepthwiseConv2D(*args, **kwargs)


class KerasDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kernel_size = kwargs.pop("kernel_size")
        padding = kwargs.pop("padding", 0)
        stride = kwargs.pop("strides", (1, 1))
        dilation = kwargs.pop("dilation_rate", (1, 1))
        data_format = kwargs.pop("data_format", "channels_last")

        self.padding_mode = kwargs.pop("padding_mode", "zeros")
        self._padding = padding
        self._previous_frame_info = None

        kernel_size_ = parse(kernel_size)
        stride_ = parse(stride)
        padding_ = padding if isinstance(padding, str) else parse(padding)
        dilation_ = parse(dilation)

        # Call the original __init__ with the remaining args and kwargs
        depth_multiplier = kwargs.pop("groups") // kwargs.pop("filters")
        self.depth_multiplier = depth_multiplier

        # pytorch layers attributes
        self.in_channels = kwargs.pop("in_channels")

        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        super().__init__(
            *args,
            kernel_size=kernel_size_,
            strides=stride_,
            dilation_rate=dilation_,
            padding="valid",
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            **kwargs,
        )

        # Compute self._reversed_padding_repeated_twice
        if isinstance(padding_, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    self.dilation_rate,
                    self.kernel_size,
                    range(len(self.kernel_size) - 1, -1, -1),
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(padding_, 2)

        depthwise_shape = self.kernel_size + (
            self.in_channels,
            self.depth_multiplier,
        )

        # create placeholder weights on initialization
        self.weight = tf.experimental.numpy.empty(
            depthwise_shape,
            dtype=tf.float32,
        )

        if self.use_bias:
            self.bias = tf.experimental.numpy.empty(
                (self.depth_multiplier * self.in_channels,),
                dtype=tf.float32,
            )
        else:
            self.bias = None

        self.v["weight"] = self.weight
        self.v["bias"] = self.bias

        os.environ["DATA_FORMAT"] = "channels_first"

    @property
    def v(self):
        return self._v

    @property
    def buffers(self):
        return self._buffers

    def named_parameters(self):
        return {k: v for k, v in self.v.items() if v is not None}

    def named_buffers(self):
        return {k: v for k, v in self.buffers.items() if v is not None}

    def eval(self):
        self.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "padding_mode": self.padding_mode,
                "kernel_size": self.kernel_size,
                "padding": self._padding,
                "strides": self.strides,
                "dilation_rate": self.dilation_rate,
                "data_format": self.data_format,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @store_frame_info
    def __call__(self, *args, **kwargs):
        if not self.built:
            res = super().__call__(*args, **kwargs)
            # recompute build shapes based on transposed input
            order = (0, 2, 3, 1)
            input_shape = args[0].shape
            new_shape = tuple(input_shape[i] for i in order)
            self._build_shapes_dict = {"input_shape": new_shape}
            return res
        return self.call(args[0])

    def __repr__(self):
        return "KerasDepthWiseConv2D()"

    def __setattr__(self, name, value):
        if name in ["_v", "_buffers"]:
            self.__dict__[name] = value
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        built = object.__getattribute__(self, "__dict__").get("built", False)

        if built:
            if parse_package(keras.__version__).major > 2:
                attr_map = {"weight": "kernel"}
            else:
                attr_map = {"weight": "depthwise_kernel"}
        else:
            attr_map = {"weight": "weight"}

        new_name = attr_map[name] if name in attr_map else name
        return super().__getattribute__(new_name)

    def build(self, input_shape):
        _, ch, _, _ = input_shape
        if (
            not self.built
            and self.data_format == "channels_last"
            and os.environ.get("DATA_FORMAT", "channels_first") == "channels_first"
        ):
            order = (0, 2, 3, 1)
            new_shape = tuple(input_shape[i] for i in order)
            input_shape = tf.TensorShape(new_shape)

        super().build(input_shape)
        # modify the channel axis to avoid shape assertion checks by keras
        self.input_spec.axes = {1: ch}
        return

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input, training=False):
        if self._padding != 0:
            padding_mode = (
                "constant" if self.padding_mode == "zeros" else self.padding_mode
            )
            # handle Pytorch-style padding
            input = torch_pad(
                input, self._reversed_padding_repeated_twice, mode=padding_mode
            )

        return super().call(input)


class KerasConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        kernel_size = kwargs.pop("kernel_size")
        padding = kwargs.pop("padding", 0)
        stride = kwargs.pop("strides", (1, 1))
        dilation = kwargs.pop("dilation_rate", (1, 1))
        data_format = kwargs.pop("data_format", "channels_last")

        self.padding_mode = kwargs.pop("padding_mode", "zeros")
        self._padding = padding
        self._previous_frame_info = None

        kernel_size_ = parse(kernel_size)
        stride_ = parse(stride)
        padding_ = padding if isinstance(padding, str) else parse(padding)
        dilation_ = parse(dilation)

        # pytorch layers attributes
        self.in_channels = kwargs.pop("in_channels")

        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        # Call the original __init__ with the remaining args and kwargs
        super().__init__(
            *args,
            kernel_size=kernel_size_,
            strides=stride_,
            dilation_rate=dilation_,
            padding="valid",
            data_format=data_format,
            **kwargs,
        )

        # Compute self._reversed_padding_repeated_twice
        if isinstance(padding_, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    self.dilation_rate,
                    self.kernel_size,
                    range(len(self.kernel_size) - 1, -1, -1),
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(padding_, 2)

        # create placeholder weights on initialization
        self.weight = tf.experimental.numpy.empty(
            (*kernel_size_, self.in_channels // kwargs["groups"], self.filters),
            dtype=tf.float32,
        )
        if self.use_bias:
            self.bias = tf.experimental.numpy.empty((self.filters,), dtype=tf.float32)
        else:
            self.bias = None

        self.v["weight"] = self.weight
        self.v["bias"] = self.bias

        os.environ["DATA_FORMAT"] = "channels_first"

    @property
    def v(self):
        return self._v

    @property
    def buffers(self):
        return self._buffers

    def named_parameters(self):
        return {k: v for k, v in self.v.items() if v is not None}

    def named_buffers(self):
        return {k: v for k, v in self.buffers.items() if v is not None}

    def eval(self):
        self.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "padding_mode": self.padding_mode,
                "kernel_size": self.kernel_size,
                "padding": self._padding,
                "strides": self.strides,
                "dilation_rate": self.dilation_rate,
                "data_format": self.data_format,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @store_frame_info
    def __call__(self, *args, **kwargs):
        if not self.built:
            res = super().__call__(*args, **kwargs)
            # recompute build shapes based on transposed input
            order = (0, 2, 3, 1)
            input_shape = args[0].shape
            new_shape = tuple(input_shape[i] for i in order)
            self._build_shapes_dict = {"input_shape": new_shape}
            return res
        return self.call(args[0])

    def __repr__(self):
        return "KerasConv2D()"

    def __setattr__(self, name, value):
        if name in ["_v", "_buffers"]:
            self.__dict__[name] = value
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        built = object.__getattribute__(self, "__dict__").get("built", False)
        if built:
            attr_map = {"weight": "kernel", "out_channels": "filters"}
        else:
            attr_map = {
                "out_channels": "filters",
            }

        new_name = attr_map[name] if name in attr_map else name
        return super().__getattribute__(new_name)

    def build(self, input_shape):
        _, ch, _, _ = input_shape
        if (
            not self.built
            and self.data_format == "channels_last"
            and os.environ.get("DATA_FORMAT", "channels_first") == "channels_first"
        ):
            order = (0, 2, 3, 1)
            new_shape = tuple(input_shape[i] for i in order)
            input_shape = tf.TensorShape(new_shape)

        super().build(input_shape)
        # modify the channel axis to avoid shape assertion checks by keras
        self.input_spec.axes = {1: ch}
        return

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input, training=False):
        if self._padding != 0:
            padding_mode = (
                "constant" if self.padding_mode == "zeros" else self.padding_mode
            )
            # handle Pytorch-style padding
            input = torch_pad(
                input, self._reversed_padding_repeated_twice, mode=padding_mode
            )
        return super().call(input)


class KerasDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None

        # pytorch layer attributes
        self.in_features = kwargs.pop("in_features")

        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        super().__init__(*args, **kwargs)

        # create placeholder weights on initialization
        self.weight = tf.experimental.numpy.empty(
            (self.units, self.in_features), dtype=tf.float32
        )
        if self.use_bias:
            self.bias = tf.experimental.numpy.empty((self.units,), dtype=tf.float32)
        else:
            self.bias = None

        self.v["weight"] = self.weight
        self.v["bias"] = self.bias

    @property
    def v(self):
        return self._v

    @property
    def buffers(self):
        return self._buffers

    def named_parameters(self):
        return {k: v for k, v in self.v.items() if v is not None}

    def named_buffers(self):
        return {k: v for k, v in self.buffers.items() if v is not None}

    def eval(self):
        self.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def __repr__(self):
        return "KerasDense()"

    def __setattr__(self, name, value):
        if name in ["_v", "_buffers"]:
            self.__dict__[name] = value
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        built = object.__getattribute__(self, "__dict__").get("built", False)
        if built:
            attr_map = {"weight": "kernel", "out_features": "units"}
        else:
            attr_map = {"out_features": "units"}
        new_name = attr_map[name] if name in attr_map else name
        return super().__getattribute__(new_name)

    def build(self, input_shape):
        super().build(input_shape)
        return

    def call(self, input, training=False):
        return super().call(input)


class KerasBatchNorm2D(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None

        # pytorch layer attributes
        self.num_features = kwargs.pop("num_features")
        self.track_running_stats = kwargs.pop("track_running_stats")

        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        super().__init__(*args, **kwargs)

        # create placeholder weights on initialization
        if self.scale:
            self.weight = tf.experimental.numpy.empty(
                (self.num_features,), dtype=tf.float32
            )
            self.bias = tf.experimental.numpy.empty(
                (self.num_features,), dtype=tf.float32
            )
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = tf.experimental.numpy.zeros(
                (self.num_features,), dtype=tf.float32
            )
            self.running_var = tf.experimental.numpy.ones(
                (self.num_features,), dtype=tf.float32
            )
            self.num_batches_tracked = tf.constant(0, dtype=tf.int64)
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

        self.v["weight"] = self.weight
        self.v["bias"] = self.bias
        self.buffers["running_mean"] = self.running_mean
        self.buffers["running_var"] = self.running_var
        self.buffers["num_batches_tracked"] = self.num_batches_tracked

        os.environ["DATA_FORMAT"] = "channels_first"

    @property
    def v(self):
        return self._v

    @property
    def buffers(self):
        return self._buffers

    def named_parameters(self):
        return {k: v for k, v in self.v.items() if v is not None}

    def named_buffers(self):
        return {k: v for k, v in self.buffers.items() if v is not None}

    def eval(self):
        self.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "track_running_stats": self.track_running_stats,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return "KerasBatchNorm2D()"

    def __setattr__(self, name, value):
        if name in ["_v", "_buffers"]:
            self.__dict__[name] = value
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        built = object.__getattribute__(self, "__dict__").get("built", False)
        if built:
            attr_map = {
                "weight": "gamma",
                "bias": "beta",
                "running_mean": "moving_mean",
                "running_var": "moving_variance",
            }
        else:
            attr_map = {}
        new_name = attr_map[name] if name in attr_map else name
        return super().__getattribute__(new_name)

    @store_frame_info
    def __call__(self, *args, **kwargs):
        if not self.built:
            res = super().__call__(*args, **kwargs)
            # recompute build shapes based on transposed input
            order = (0, 2, 3, 1)
            input_shape = args[0].shape
            new_shape = tuple(input_shape[i] for i in order)
            self._build_shapes_dict = {"input_shape": new_shape}
            return res
        return self.call(args[0])

    def build(self, input_shape):
        _, ch, _, _ = input_shape
        if (
            not self.built
            and self.axis == -1
            and os.environ.get("DATA_FORMAT", "channels_first") == "channels_first"
        ):
            order = (0, 2, 3, 1)
            new_shape = tuple(input_shape[i] for i in order)
            input_shape = tf.TensorShape(new_shape)

        super().build(input_shape)
        # modify the channel axis to avoid shape assertion checks by keras
        self.input_spec.axes = {1: ch}
        return

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input, training=False):
        return super().call(input, training=training)


class KerasReLU(tf.keras.layers.ReLU):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "KerasReLU()"

    @store_frame_info
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input, training=False):
        return super().call(input)
