import textwrap


KerasNativeLayers = textwrap.dedent(
    """
from .ivy.func_wrapper import tensorflow_handle_array_like_without_promotion
from .tensorflow__stateful import store_frame_info
from .ivy.utils.decorator_utils import (
    tensorflow_handle_transpose_in_input_and_output,
)
import tensorflow as tf
import keras
import collections
from itertools import repeat
from numbers import Number
import os
from packaging.version import parse as parse_package

if parse_package(keras.__version__).major > 2:
    KerasVariable = keras.src.backend.Variable
else:
    KerasVariable = tf.Variable


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
        self._previous_frame_info = None
        self._is_built = False

        if "kernel_size" in kwargs:
            kernel_size = kwargs.pop("kernel_size")
        else:
            kernel_size = args[2]
        padding = kwargs.pop("padding", 0)
        stride = kwargs.pop("strides", (1, 1))
        dilation = kwargs.pop("dilation_rate", (1, 1))
        data_format = kwargs.pop("data_format", "channels_last")

        self.padding_mode = kwargs.pop("padding_mode", "zeros")
        self._padding = padding

        kernel_size_ = parse(kernel_size)
        stride_ = parse(stride)
        padding_ = padding if isinstance(padding, str) else parse(padding)
        dilation_ = parse(dilation)

        # Call the original __init__ with the remaining args and kwargs
        depth_multiplier = kwargs.pop("groups", 1) // (
            kwargs.pop("filters") if "filters" in kwargs else args[1]
        )
        self.depth_multiplier = depth_multiplier

        # pytorch layers attributes
        if "in_channels" in kwargs:
            self.in_channels = kwargs.pop("in_channels")
        else:
            self.in_channels = args[0]

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

        # create PT placeholder weights on initialization
        self.pt_weight = tf.experimental.numpy.empty(
            depthwise_shape,
            dtype=tf.float32,
        )

        if self.use_bias:
            self.pt_bias = tf.experimental.numpy.empty(
                (self.depth_multiplier * self.in_channels,),
                dtype=tf.float32,
            )
        else:
            self.pt_bias = None

        self.v["weight"] = self.pt_weight
        self.v["bias"] = self.pt_bias

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

    def named_children(self):
        return {}

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        memo.add(self)
        return [(prefix, self)]

    def children(self):
        return []
    
    def parameters(self):
        return self.named_parameters().values()

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
        elif name in ["weight", "bias"] and hasattr(self, name):
            new_value = value
            if value is not None:
                # Determine the transpose type based on the value shape
                if len(value.shape) > 3:
                    transpose_axes = (2, 3, 0, 1)  # DepthwiseConv2D transpose axes
                else:
                    transpose_axes = None  # No transposition needed for other cases

                # Apply transpose if required
                if transpose_axes is not None:
                    new_value = tf.transpose(value, transpose_axes)

                if not isinstance(new_value, KerasVariable):
                    new_value = KerasVariable(new_value)

            if parse_package(keras.__version__).major > 2:
                new_native_name = "kernel" if name == "weight" else name
            else:
                new_native_name = "depthwise_kernel" if name == "weight" else name
            try:
                weight_var = getattr(self, new_native_name)
                if isinstance(weight_var, KerasVariable):
                    weight_var.assign(new_value)
                else:
                    # using object.__setattr__ to avoid Keras internally creating new parameters.
                    object.__setattr__(self, new_native_name, new_value)
            except AttributeError:
                # using object.__setattr__ to avoid Keras internally creating new parameters.
                object.__setattr__(self, new_native_name, new_value)

            new_pt_name = "pt_weight" if name == "weight" else "pt_bias"
            object.__setattr__(self, new_pt_name, new_value)

        elif name in ["pt_weight", "pt_bias"] and hasattr(self, name):
            new_value = value
            if value is not None:
                # Determine the transpose type based on the value shape
                if len(value.shape) > 3:
                    transpose_axes = (2, 3, 0, 1)  # DepthwiseConv2D transpose axes
                else:
                    transpose_axes = None  # No transposition needed for other cases

                # Apply transpose if required
                if transpose_axes is not None:
                    new_value = tf.transpose(value, transpose_axes)
            object.__setattr__(self, name, new_value)
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        is_built = object.__getattribute__(self, "__dict__").get("_is_built", False)
        built = object.__getattribute__(self, "__dict__").get("built", False)
        use_bias = object.__getattribute__(self, "__dict__").get("use_bias", True)
        if built and is_built:
            if parse_package(keras.__version__).major > 2:
                attr_map = {"weight": "kernel", "bias": "bias"}
            else:
                attr_map = {"weight": "depthwise_kernel", "bias": "bias"}
        else:
            attr_map = {"weight": "pt_weight", "bias": "pt_bias"}
        if not use_bias:
            attr_map["bias"] = "pt_bias"
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
        self._is_built = True
        res = super().call(input)
        self._is_built = False
        return res

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self, "children"):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self

    def to(self, *args, **kwargs):
        from .ivy.functional.frontends.torch.tensor import tensorflow_to_frnt_

        return self._apply(lambda t: tensorflow_to_frnt_(t, *args, **kwargs))


class KerasConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None
        self._is_built = False

        # pytorch layers attributes
        self.in_channels = kwargs.pop("in_channels")
        self.out_channels = kwargs["filters"]
        kernel_size = kwargs.pop("kernel_size")
        padding = kwargs.pop("padding", 0)
        stride = kwargs.pop("strides", (1, 1))
        dilation = kwargs.pop("dilation_rate", (1, 1))
        data_format = kwargs.pop("data_format", "channels_last")

        self.padding_mode = kwargs.pop("padding_mode", "zeros")
        self._padding = padding

        kernel_size_ = parse(kernel_size)
        stride_ = parse(stride)
        padding_ = padding if isinstance(padding, str) else parse(padding)
        dilation_ = parse(dilation)

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

        # create PT placeholder weights on initialization
        self.pt_weight = tf.experimental.numpy.empty(
            (self.out_channels, self.in_channels // kwargs["groups"], *kernel_size_),
            dtype=tf.float32,
        )
        if self.use_bias:
            self.pt_bias = tf.experimental.numpy.empty(
                (self.out_channels,), dtype=tf.float32
            )
        else:
            self.pt_bias = None

        self.v["weight"] = self.pt_weight
        self.v["bias"] = self.pt_bias

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

    def named_children(self):
        return {}

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        memo.add(self)
        return [(prefix, self)]

    def children(self):
        return []
    
    def parameters(self):
        return self.named_parameters().values()
        
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
        elif name in ["weight", "bias"] and hasattr(self, name):
            new_value = value
            if value is not None:
                # Determine the transpose type based on the value shape
                if len(value.shape) > 4:
                    # Conv3D case: PT [out_channels, in_channels, depth, height, width]
                    # TF/JAX [depth, height, width, in_channels, out_channels]
                    transpose_axes = (2, 3, 4, 1, 0)  # Conv3D transpose axes
                elif len(value.shape) > 3:
                    # Conv2D case: PT [out_channels, in_channels, height, width]
                    # TF/JAX [height, width, in_channels, out_channels]
                    transpose_axes = (2, 3, 1, 0)  # Conv2D transpose axes
                elif len(value.shape) > 2:
                    # Conv1D case: PT [out_channels, in_channels, length]
                    # TF/JAX [length, in_channels, out_channels]
                    transpose_axes = (2, 1, 0)  # Conv1D transpose axes
                else:
                    transpose_axes = None  # No transposition needed for other cases

                # Apply transpose if required
                if transpose_axes is not None:
                    new_value = tf.transpose(value, transpose_axes)

                if not isinstance(new_value, KerasVariable):
                    new_value = KerasVariable(new_value)

            new_native_name = "_kernel" if name == "weight" else name
            try:
                weight_var = getattr(self, new_native_name)
                if isinstance(weight_var, KerasVariable):
                    weight_var.assign(new_value)
                else:
                    # using object.__setattr__ to avoid Keras internally creating new parameters.
                    object.__setattr__(self, new_native_name, new_value)
            except AttributeError:
                # using object.__setattr__ to avoid Keras internally creating new parameters.
                object.__setattr__(self, new_native_name, new_value)

            new_pt_name = "pt_weight" if name == "weight" else "pt_bias"
            object.__setattr__(self, new_pt_name, value)
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        is_built = object.__getattribute__(self, "__dict__").get("_is_built", False)
        built = object.__getattribute__(self, "__dict__").get("built", False)
        use_bias = object.__getattribute__(self, "__dict__").get("use_bias", True)
        if built and is_built:
            attr_map = {"weight": "_kernel", "bias": "bias"}
        else:
            attr_map = {"weight": "pt_weight", "bias": "pt_bias"}
        if not use_bias:
            attr_map["bias"] = "pt_bias"
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
        self._is_built = True
        if self._padding != 0:
            padding_mode = (
                "constant" if self.padding_mode == "zeros" else self.padding_mode
            )
            # handle Pytorch-style padding
            input = torch_pad(
                input, self._reversed_padding_repeated_twice, mode=padding_mode
            )
        res = super().call(input)
        self._is_built = False
        return res

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self, "children"):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self

    def to(self, *args, **kwargs):
        from .ivy.functional.frontends.torch.tensor import tensorflow_to_frnt_

        return self._apply(lambda t: tensorflow_to_frnt_(t, *args, **kwargs))


class KerasDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None
        self._is_built = False

        # pytorch layer attributes
        self.in_features = kwargs.pop("in_features")
        self.out_features = kwargs.pop("units")
        use_bias = kwargs.pop("use_bias", True)

        super().__init__(
            units=self.out_features,
            use_bias=use_bias,
            *args,
            **kwargs,
        )
        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        # create PT placeholder weights on initialization
        self.pt_weight = tf.experimental.numpy.empty(
            (self.out_features, self.in_features), dtype=tf.float32
        )
        if self.use_bias:
            self.pt_bias = tf.experimental.numpy.empty(
                (self.out_features,), dtype=tf.float32
            )
        else:
            self.pt_bias = None

        self.v["weight"] = self.pt_weight
        self.v["bias"] = self.pt_bias

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

    def named_children(self):
        return {}

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        memo.add(self)
        return [(prefix, self)]

    def children(self):
        return []
    
    def parameters(self):
        return self.named_parameters().values()

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
        elif name in ["weight", "bias"] and hasattr(self, name):
            new_native_name = "_kernel" if name == "weight" else name
            new_value = tf.transpose(value, (1, 0)) if name == "weight" else value
            if new_value is not None:
                if not isinstance(new_value, KerasVariable):
                    new_value = KerasVariable(new_value)

                try:
                    weight_var = getattr(self, new_native_name)
                    if isinstance(weight_var, KerasVariable):
                        weight_var.assign(new_value)
                    else:
                        # using object.__setattr__ to avoid Keras internally creating new parameters.
                        object.__setattr__(self, new_native_name, new_value)
                except AttributeError:
                    # using object.__setattr__ to avoid Keras internally creating new parameters.
                    object.__setattr__(self, new_native_name, new_value)

            new_pt_name = "pt_weight" if name == "weight" else "pt_bias"
            object.__setattr__(self, new_pt_name, value)
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        is_built = object.__getattribute__(self, "__dict__").get("_is_built", False)
        built = object.__getattribute__(self, "__dict__").get("built", False)
        use_bias = object.__getattribute__(self, "__dict__").get("use_bias", True)
        if built and is_built:
            attr_map = {"weight": "_kernel", "bias": "bias"}
        else:
            attr_map = {"weight": "pt_weight", "bias": "pt_bias"}
        if not use_bias:
            attr_map["bias"] = "pt_bias"
        new_name = attr_map[name] if name in attr_map else name
        return super().__getattribute__(new_name)

    def build(self, input_shape):
        super().build(input_shape)
        return

    def call(self, input, training=False):
        self._is_built = True
        res = super().call(input)
        self._is_built = False
        return res

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self, "children"):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self

    def to(self, *args, **kwargs):
        from .ivy.functional.frontends.torch.tensor import tensorflow_to_frnt_

        return self._apply(lambda t: tensorflow_to_frnt_(t, *args, **kwargs))


class KerasBatchNorm2D(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None
        self._is_built = False

        # pytorch layer attributes
        if "num_features" in kwargs:
            self.num_features = kwargs.pop("num_features")
        else:
            self.num_features = args[0]
        self.track_running_stats = kwargs.pop("track_running_stats", True)

        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        super().__init__(*args, **kwargs)

        # create PT placeholder weights on initialization
        if self.scale:
            self.pt_weight = tf.experimental.numpy.empty(
                (self.num_features,), dtype=tf.float32
            )
            self.pt_bias = tf.experimental.numpy.empty(
                (self.num_features,), dtype=tf.float32
            )
        else:
            self.pt_weight = None
            self.pt_bias = None

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

        self.v["weight"] = self.pt_weight
        self.v["bias"] = self.pt_bias
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

    def named_children(self):
        return {}

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        memo.add(self)
        return [(prefix, self)]

    def children(self):
        return []
    
    def parameters(self):
        return self.named_parameters().values()

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
        elif name in ["weight", "bias"] and hasattr(self, name):
            new_native_name = "gamma" if name == "weight" else "beta"
            if value is not None:
                if not isinstance(value, KerasVariable):
                    new_value = KerasVariable(value)
                else:
                    new_value = value
            try:
                weight_var = getattr(self, new_native_name)
                if isinstance(weight_var, KerasVariable):
                    weight_var.assign(new_value)
                else:
                    # using object.__setattr__ to avoid Keras internally creating new parameters.
                    object.__setattr__(self, new_native_name, new_value)
            except AttributeError:
                # using object.__setattr__ to avoid Keras internally creating new parameters.
                object.__setattr__(self, new_native_name, new_value)

            new_pt_name = "pt_weight" if name == "weight" else "pt_bias"
            object.__setattr__(self, new_pt_name, value)
            return
        elif name in ["running_mean", "running_var"] and hasattr(self, name):
            if value is not None:
                if not isinstance(value, KerasVariable):
                    new_value = KerasVariable(value)
                else:
                    new_value = value
            new_native_name = (
                "moving_mean" if name == "running_mean" else "moving_variance"
            )
            try:
                weight_var = getattr(self, new_native_name)
                if isinstance(weight_var, KerasVariable):
                    weight_var.assign(new_value)
                else:
                    # using object.__setattr__ to avoid Keras internally creating new parameters.
                    object.__setattr__(self, new_native_name, new_value)
            except AttributeError:
                # using object.__setattr__ to avoid Keras internally creating new parameters.
                object.__setattr__(self, new_native_name, new_value)

            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        is_built = object.__getattribute__(self, "__dict__").get("_is_built", False)
        built = object.__getattribute__(self, "__dict__").get("built", False)
        _scale = object.__getattribute__(self, "__dict__").get("scale", False)
        _center = object.__getattribute__(self, "__dict__").get("center", False)
        if built and is_built:
            attr_map = {
                "weight": "gamma",
                "bias": "beta",
                "running_mean": "moving_mean",
                "running_var": "moving_variance",
            }
        else:
            attr_map = {
                "weight": "pt_weight",
                "bias": "pt_bias",
                "running_mean": "running_mean",
                "running_var": "running_var",
            }
        if not _scale:
            attr_map["weight"] = "pt_weight"
        if not _center:
            attr_map["bias"] = "pt_bias"
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
        self._is_built = True
        res = super().call(input, training=training)
        self._is_built = False
        return res

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self, "children"):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self

    def to(self, *args, **kwargs):
        from .ivy.functional.frontends.torch.tensor import tensorflow_to_frnt_

        return self._apply(lambda t: tensorflow_to_frnt_(t, *args, **kwargs))


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

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self, "children"):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self

    def to(self, *args, **kwargs):
        from .ivy.functional.frontends.torch.tensor import tensorflow_to_frnt_

        return self._apply(lambda t: tensorflow_to_frnt_(t, *args, **kwargs))
"""
)


FlaxNativeLayers = textwrap.dedent(
    """
from typing import Tuple, Union, Optional, Any, Callable, Iterable, Literal
from numbers import Number
import collections.abc
import os
import jax
import jax.numpy as jnp
import jax.lax as jlax
from flax import nnx

from .jax__stateful import store_frame_info
from .ivy.utils.decorator_utils import (
    jax_handle_transpose_in_input_and_output,
)
from .ivy.functional.ivy.general import jax_is_array_bknd


def parse(x):
    n = 2
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(x for _ in range(n))


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


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def _to_nested_tuple(nested_list):
    ret = ()
    if hasattr(nested_list, "__iter__"):
        for inner_list in nested_list:
            if hasattr(inner_list, "__iter__"):
                ret += (tuple(inner_list),)
            else:
                ret += (inner_list,)
        return ret
    if ret == ():
        return nested_list


def jax_pad(
    input: jax.Array,
    pad_width: Union[Iterable[Tuple[int]], int],
    /,
    *,
    mode: Union[
        Literal[
            "constant",
            "dilated",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
            "empty",
        ],
        Callable,
    ] = "constant",
    stat_length: Union[Iterable[Tuple[int]], int] = 1,
    constant_values: Union[Iterable[Tuple[Number]], Number] = 0,
    end_values: Union[Iterable[Tuple[Number]], Number] = 0,
    reflect_type: Literal["even", "odd"] = "even",
    **kwargs: Optional[Any],
) -> jax.Array:
    pad_width = _to_nested_tuple(pad_width)
    stat_length = _to_nested_tuple(stat_length)
    constant_values = _to_nested_tuple(constant_values)
    end_values = _to_nested_tuple(end_values)
    input_dtype = input.dtype

    if mode == "dilated":
        if (
            not jax_is_array_bknd(constant_values)
            or constant_values.dtype != input_dtype
        ):
            constant_values = jnp.array(constant_values, dtype=input_dtype)
        return jlax.pad(input, constant_values, pad_width)
    if callable(mode):
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            **kwargs,
        )
    elif mode in ["maximum", "mean", "median", "minimum"]:
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            stat_length=stat_length,
        )
    elif mode == "constant":
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            constant_values=constant_values,
        )
    elif mode == "linear_ramp":
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            end_values=end_values,
        )
    elif mode in ["reflect", "symmetric"]:
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            reflect_type=reflect_type,
        )
    else:
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
        )
    if jnp.issubdtype(input_dtype, jnp.integer) and mode in ["mean", "median"]:
        ret = jnp.astype(jnp.round(ret), input_dtype)
    return ret


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
    order = (0, 2, 3, 1)
    pad = tuple(pad[i] for i in order)
    return jax_pad(input, pad, mode=mode_dict[mode], constant_values=value)


class FlaxConv(nnx.Conv):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None
        self._built = False

        kernel_size = kwargs.pop("kernel_size", (3, 3))
        padding = kwargs.pop("padding", 0)
        stride = kwargs.pop("strides", (1, 1))
        dilation = kwargs.pop("kernel_dilation", (1, 1))

        self.padding_mode = kwargs.pop("padding_mode", "zeros")
        self._padding = padding

        kernel_size_ = parse(kernel_size)
        stride_ = parse(stride)
        padding_ = padding if isinstance(padding, str) else parse(padding)
        dilation_ = parse(dilation)

        in_features = kwargs.pop("in_features")
        out_features = kwargs.pop("out_features")

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size_,
            strides=stride_,
            kernel_dilation=dilation_,
            padding=padding,
            rngs=nnx.Rngs(0),
            *args,
            **kwargs,
        )

        # Compute self._reversed_padding_repeated_twice
        if isinstance(padding_, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if padding_ == "same":
                for d, k, i in zip(
                    self.kernel_dilation,
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

        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        # create PT style placeholder weights on initialization
        self.pt_weight = jnp.empty(
            (out_features, in_features // kwargs["feature_group_count"], *kernel_size_),
            dtype=jnp.float32,
        )
        if self.use_bias:
            self.pt_bias = jnp.empty((out_features,), dtype=jnp.float32)
        else:
            self.pt_bias = None

        self.v["weight"] = self.kernel
        self.v["bias"] = self.bias

        os.environ["DATA_FORMAT"] = "channels_first"

    @store_frame_info
    @jax_handle_transpose_in_input_and_output
    def __call__(self, inputs):
        self._built = True
        if self.padding_mode != "zeros":
            padding_mode = (
                "constant" if self.padding_mode == "zeros" else self.padding_mode
            )
            # handle Pytorch-style padding
            inputs = torch_pad(
                inputs, self._reversed_padding_repeated_twice, mode=padding_mode
            )
            old_pad = self.padding
            self.padding = 0
            logits = super().__call__(inputs)
            self.padding = old_pad
            self._built = False
            return logits
        logits = super().__call__(inputs)
        self._built = False
        return logits

    def __repr__(self):
        return f"FlaxConv(in_features={self.in_features}, out_features={self.out_features}, kernel_size={self.kernel_size}, strides={self.strides}, padding={self._padding}, padding_mode={self.padding_mode})"

    def __setattr__(self, name, value):
        if name in ["_v", "_buffers"]:
            self.__dict__[name] = value
            return
        elif name in ["weight", "bias"] and hasattr(self, name):
            # Determine the transpose type based on the value shape
            if len(value.shape) > 4:
                # Conv3D case: PT [out_channels, in_channels, depth, height, width]
                # TF/JAX [depth, height, width, in_channels, out_channels]
                transpose_axes = (2, 3, 4, 1, 0)  # Conv3D transpose axes
            elif len(value.shape) > 3:
                # Conv2D case: PT [out_channels, in_channels, height, width]
                # TF/JAX [height, width, in_channels, out_channels]
                transpose_axes = (2, 3, 1, 0)  # Conv2D transpose axes
            elif len(value.shape) > 2:
                # Conv1D case: PT [out_channels, in_channels, length]
                # TF/JAX [length, in_channels, out_channels]
                transpose_axes = (2, 1, 0)  # Conv1D transpose axes
            else:
                transpose_axes = None  # No transposition needed for other cases

            # Apply transpose if required
            if transpose_axes is not None:
                new_value = jnp.transpose(value, transpose_axes)
            else:
                new_value = value

            if not isinstance(new_value, nnx.Param):
                new_value = nnx.Param(new_value)
            new_native_name = "kernel" if name == "weight" else name
            object.__setattr__(self, new_native_name, new_value)

            new_pt_name = "pt_weight" if name == "weight" else "pt_bias"
            object.__setattr__(self, new_pt_name, value)
            return
        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        built = object.__getattribute__(self, "__dict__").get("_built", False)
        if built:
            attr_map = {
                "weight": "kernel",
                "bias": "bias",
                "in_channels": "in_features",
                "out_channels": "out_features",
                "stride": "strides",
                "groups": "feature_group_count",
                "dilation": "input_dilation",
                "dilation": "kernel_dilation",
            }
        else:
            attr_map = {
                "weight": "pt_weight",
                "bias": "pt_bias",
                "in_channels": "in_features",
                "out_channels": "out_features",
                "stride": "strides",
                "groups": "feature_group_count",
                "dilation": "input_dilation",
                "dilation": "kernel_dilation",
            }
        new_name = attr_map[name] if name in attr_map else name
        return super().__getattribute__(new_name)

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

    def named_children(self):
        return {}

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        memo.add(self)
        return [(prefix, self)]

    def children(self):
        return []
    
    def parameters(self):
        return self.named_parameters().values()

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self,'children'):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self
    
    def to(self, *args, **kwargs):
        from .ivy.functional.frontends.torch.tensor import jax_to_frnt_

        return self._apply(lambda t: jax_to_frnt_(t, *args, **kwargs))

class FlaxBatchNorm(nnx.BatchNorm):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None
        self._built = False

        # Map PyTorch-style parameters to Flax parameters
        self.num_batches_tracked = jnp.array(0)
        self.track_running_stats = kwargs.pop("track_running_stats", True)

        num_features = kwargs.pop("num_features")
        momentum = kwargs.pop("momentum", 0.1)
        epsilon = kwargs.pop("epsilon", 1e-5)
        use_bias = kwargs.pop("use_bias", True)
        use_scale = kwargs.pop("use_scale", True)

        super().__init__(
            num_features=num_features,
            use_running_average=True,
            momentum=1 - momentum,  # Flax uses decay rate, while PyTorch uses momentum
            epsilon=epsilon,
            use_bias=use_bias,
            use_scale=use_scale,
            rngs=nnx.Rngs(0),
            *args,
            **kwargs,
        )

        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        # create PT style placeholder weights on initialization
        if self.scale.value is not None:
            self.pt_weight = jnp.empty((num_features,), dtype=jnp.float32)
            self.pt_bias = jnp.empty((num_features,), dtype=jnp.float32)
        else:
            self.pt_weight = None
            self.pt_bias = None

        if self.track_running_stats:
            self.running_mean = jnp.zeros((num_features,), dtype=jnp.float32)
            self.running_var = jnp.ones((num_features,), dtype=jnp.float32)
            self.num_batches_tracked = jnp.asarray(0, dtype=jnp.int64)
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

        self.v["weight"] = self.scale
        self.v["bias"] = self.bias
        self.buffers["running_mean"] = self.mean
        self.buffers["running_var"] = self.var
        self.buffers["num_batches_tracked"] = self.num_batches_tracked
        # Set data format for compatibility
        os.environ["DATA_FORMAT"] = "channels_first"

    @store_frame_info
    @jax_handle_transpose_in_input_and_output
    def __call__(self, inputs, use_running_average=None, *, mask=None):
        self._built = True
        logits = super().__call__(
            inputs, use_running_average=use_running_average, mask=mask
        )
        self._built = False
        return logits

    def __repr__(self):
        return (
            f"FlaxBatchNorm2D({self.num_features}, "
            f"eps={self.epsilon}, momentum={1-self.momentum}, "
            f"affine={self.use_scale}, "
        )

    def __setattr__(self, name, value):
        if name in ["_v", "_buffers"]:
            self.__dict__[name] = value
            return
        elif name in ["weight", "bias"] and hasattr(self, name):
            new_native_name = "scale" if name == "weight" else name
            if not isinstance(value, nnx.Param):
                new_value = nnx.Param(value)
            else:
                new_value = value
            object.__setattr__(self, new_native_name, new_value)

            new_pt_name = "pt_weight" if name == "weight" else "pt_bias"
            object.__setattr__(self, new_pt_name, value)
            return
        elif name in ["running_mean", "running_var"] and hasattr(self, name):
            if not isinstance(value, nnx.BatchStat):
                new_value = nnx.BatchStat(value)
            else:
                new_value = value
            new_native_name = "mean" if name == "running_mean" else "var"
            object.__setattr__(self, new_native_name, new_value)

        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        built = object.__getattribute__(self, "__dict__").get("_built", False)
        if built:
            attr_map = {
                "weight": "scale",
                "bias": "bias",
                "eps": "epsilon",
                "affine": "use_scale",
                "running_mean": "mean",
                "running_var": "var",
            }
        else:
            attr_map = {
                "weight": "pt_weight",
                "bias": "pt_bias",
                "eps": "epsilon",
                "affine": "use_scale",
            }
        new_name = attr_map[name] if name in attr_map else name
        return super().__getattribute__(new_name)

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

    def named_children(self):
        return {}

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        memo.add(self)
        return [(prefix, self)]

    def children(self):
        return []
    
    def parameters(self):
        return self.named_parameters().values()

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self,'children'):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self
    
    def to(self, *args, **kwargs):
        from .ivy.functional.frontends.torch.tensor import jax_to_frnt_

        return self._apply(lambda t: jax_to_frnt_(t, *args, **kwargs))

class FlaxLinear(nnx.Linear):
    def __init__(self, *args, **kwargs):
        self._previous_frame_info = None
        self._built = False

        in_features = kwargs.pop("in_features")
        out_features = kwargs.pop("out_features")
        use_bias = kwargs.pop("use_bias", True)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            rngs=nnx.Rngs(0),
            *args,
            **kwargs,
        )

        # ivy.Module attributes
        self._v = dict()
        self._buffers = dict()

        # create PT style placeholder weights on initialization
        self.pt_weight = jnp.empty((out_features, in_features), dtype=jnp.float32)
        if self.use_bias:
            self.pt_bias = jnp.empty((out_features,), dtype=jnp.float32)
        else:
            self.pt_bias = None

        self.v["weight"] = self.kernel
        self.v["bias"] = self.bias

    def __call__(self, inputs):
        self._built = True
        logits = super().__call__(inputs)
        self._built = False
        return logits

    def __repr__(self):
        return f"FlaxLinear(in_features={self.in_features}, out_features={self.out_features}, use_bias={self.use_bias})"

    def __setattr__(self, name, value):
        if name in ["_v", "_buffers"]:
            self.__dict__[name] = value
            return
        elif name in ["weight", "bias"] and hasattr(self, name):
            new_native_name = "kernel" if name == "weight" else name
            new_value = jnp.transpose(value, (1, 0)) if name == "weight" else value
            if not isinstance(new_value, nnx.Param):
                new_value = nnx.Param(new_value)
            object.__setattr__(self, new_native_name, new_value)

            new_pt_name = "pt_weight" if name == "weight" else "pt_bias"
            object.__setattr__(self, new_pt_name, value)
            return
        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        built = object.__getattribute__(self, "__dict__").get("_built", False)
        if built:
            attr_map = {"weight": "kernel", "bias": "bias"}
        else:
            attr_map = {"weight": "pt_weight", "bias": "pt_bias"}
        new_name = attr_map[name] if name in attr_map else name
        return super().__getattribute__(new_name)

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

    def named_children(self):
        return {}

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        memo.add(self)
        return [(prefix, self)]

    def children(self):
        return []
    
    def parameters(self):
        return self.named_parameters().values()

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self,'children'):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self
    
    def to(self, *args, **kwargs):
        from .ivy.functional.frontends.torch.tensor import jax_to_frnt_

        return self._apply(lambda t: jax_to_frnt_(t, *args, **kwargs))
"""
)
