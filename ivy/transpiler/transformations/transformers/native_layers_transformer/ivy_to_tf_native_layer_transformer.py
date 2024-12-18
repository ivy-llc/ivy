import gast
from .base_transformer import (
    PytorchToFrameworkLayer,
)


class PytorchToKerasLayer(PytorchToFrameworkLayer):
    def get_layer_mapping(self):
        return {
            "tensorflow_Conv2d": self.convert_conv2d,
            "tensorflow_Linear": self.convert_linear,
            "tensorflow_BatchNorm2d": self.convert_batchnorm2d,
            # Add more mappings here...
        }

    def get_name_mapping(self):
        return {
            "tensorflow_Conv2d": ("KerasConv2D", self.convert_conv2d),
            "tensorflow_Linear": ("KerasDense", self.convert_linear),
            "tensorflow_BatchNorm2d": ("KerasBatchNorm2D", self.convert_batchnorm2d),
            # Add more mappings here...
        }

    def convert_conv2d(self, node, is_alias=False):
        # argument order for nn.Conv2d
        conv2d_args = [
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "bias",
            "padding_mode",
            "device",
            "dtype",
        ]
        # Convert all args to kwargs based on the input signature of nn.Conv2d
        args_kwargs = self.convert_args_to_kwargs(node, conv2d_args)

        # Convert PyTorch Conv2d to Keras Conv2D
        kwargs = {
            "in_channels": args_kwargs["in_channels"],
            "filters": args_kwargs["out_channels"],
            "kernel_size": args_kwargs["kernel_size"],
            "strides": args_kwargs.get("stride", gast.Constant(value=1, kind=None)),
            "use_bias": args_kwargs.get("bias", gast.Constant(value=True, kind=None)),
            "dilation_rate": args_kwargs.get(
                "dilation", gast.Constant(value=1, kind=None)
            ),
            "groups": args_kwargs.get("groups", gast.Constant(value=1, kind=None)),
            "padding": args_kwargs.get("padding", gast.Constant(value=0, kind=None)),
            "padding_mode": args_kwargs.get(
                "padding_mode", gast.Constant(value="zeros", kind=None)
            ),
            "data_format": args_kwargs.get(
                "data_format", gast.Constant(value=self.data_format, kind=None)
            ),
        }

        if isinstance(kwargs["groups"], gast.Name):
            layer_name = "resolve_convolution"
        elif (
            isinstance(kwargs["groups"], gast.Constant) and kwargs["groups"].value != 1
        ):
            layer_name = "KerasDepthwiseConv2D"
        else:
            layer_name = "KerasConv2D"

        self.create_local_import(node, layer_name)
        return gast.Call(
            func=(
                node.func
                if is_alias
                else gast.Name(id=layer_name, ctx=gast.Load(), annotation=None)
            ),
            args=[],
            keywords=[
                gast.keyword(arg=key, value=value) for key, value in kwargs.items()
            ],
        )

    def convert_linear(self, node, is_alias=False):
        # argument order for nn.Linear
        linear_args = ["in_features", "out_features", "bias"]

        # Convert all args to kwargs based on the input signature of nn.Linear
        args_kwargs = self.convert_args_to_kwargs(node, linear_args)

        # Convert PyTorch Linear to Keras Dense
        kwargs = {
            "in_features": args_kwargs["in_features"],
            "units": args_kwargs["out_features"],
            "use_bias": args_kwargs.get("bias", gast.Constant(value=True, kind=None)),
        }

        self.create_local_import(node, "KerasDense")

        return gast.Call(
            func=(
                node.func
                if is_alias
                else gast.Name(id="KerasDense", ctx=gast.Load(), annotation=None)
            ),
            args=[],
            keywords=[
                gast.keyword(arg=key, value=value) for key, value in kwargs.items()
            ],
        )

    def convert_batchnorm2d(self, node, is_alias=False):
        # argument order for nn.BatchNorm2d
        batchnorm_args = [
            "num_features",
            "eps",
            "momentum",
            "affine",
            "track_running_stats",
        ]

        # Convert all args to kwargs based on the input signature of nn.BatchNorm2d
        args_kwargs = self.convert_args_to_kwargs(node, batchnorm_args)
        axis = 1 if self.data_format == "channels_first" else -1
        # Convert PyTorch BatchNorm2d to Keras BatchNormalization
        kwargs = {
            "num_features": args_kwargs.get("num_features"),
            "momentum": args_kwargs.get(
                "momentum", gast.Constant(value=0.1, kind=None)
            ),
            "epsilon": args_kwargs.get("eps", gast.Constant(value=1e-05, kind=None)),
            "center": args_kwargs.get("affine", gast.Constant(value=True, kind=None)),
            "scale": args_kwargs.get("affine", gast.Constant(value=True, kind=None)),
            "axis": args_kwargs.get("axis", gast.Constant(value=axis, kind=None)),
            "track_running_stats": args_kwargs.get(
                "track_running_stats", gast.Constant(value=True, kind=None)
            ),
        }

        self.create_local_import(node, "KerasBatchNorm2D")

        return gast.Call(
            func=(
                node.func
                if is_alias
                else gast.Name(id="KerasBatchNorm2D", ctx=gast.Load(), annotation=None)
            ),
            args=[],
            keywords=[
                gast.keyword(arg=key, value=value) for key, value in kwargs.items()
            ],
        )

    def convert_maxpool(self, node, is_alias=False):
        # argument order for nn.MaxPool2d
        maxpool_args = [
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "return_indices",
            "ceil_mode",
        ]

        # Convert all args to kwargs based on the input signature of nn.MaxPool2d
        args_kwargs = self.convert_args_to_kwargs(node, maxpool_args)

        # Convert PyTorch MaxPool2d to Keras MaxPooling2D
        kwargs = {
            "pool_size": args_kwargs["kernel_size"],
            "strides": args_kwargs.get("stride", gast.Constant(value=1, kind=None)),
            "padding": args_kwargs.get("padding", gast.Constant(value=0, kind=None)),
            "data_format": args_kwargs.get(
                "data_format", gast.Constant(value=self.data_format, kind=None)
            ),
        }

        self.create_local_import(node, "KerasMaxPooling2D")

        return gast.Call(
            func=(
                node.func
                if is_alias
                else gast.Name(id="KerasMaxPooling2D", ctx=gast.Load(), annotation=None)
            ),
            args=[],
            keywords=[
                gast.keyword(arg=key, value=value) for key, value in kwargs.items()
            ],
        )

    def get_import_module(self):
        return "tensorflow__stateful_layers"
