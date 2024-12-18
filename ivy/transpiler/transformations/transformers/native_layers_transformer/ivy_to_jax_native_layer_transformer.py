import gast
from .base_transformer import (
    PytorchToFrameworkLayer,
)


class PytorchToFlaxLayer(PytorchToFrameworkLayer):
    def get_layer_mapping(self):
        return {
            "jax_Conv2d": self.convert_conv2d,
            "jax_Linear": self.convert_linear,
            "jax_BatchNorm2d": self.convert_batchnorm2d,
            # Add more mappings here...
        }

    def get_name_mapping(self):
        return {
            "jax_Conv2d": ("FlaxConv", self.convert_conv2d),
            "jax_Linear": ("FlaxLinear", self.convert_linear),
            "jax_BatchNorm2d": ("FlaxBatchNorm", self.convert_batchnorm2d),
            # Add more mappings here...
        }

    def convert_conv2d(self, node, is_alias=False):
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
        args_kwargs = self.convert_args_to_kwargs(node, conv2d_args)

        kwargs = {
            "in_features": args_kwargs["in_channels"],
            "out_features": args_kwargs["out_channels"],
            "kernel_size": args_kwargs["kernel_size"],
            "strides": args_kwargs.get("stride", gast.Constant(value=1, kind=None)),
            "padding": args_kwargs.get("padding", gast.Constant(value=0, kind=None)),
            "padding_mode": args_kwargs.get(
                "padding_mode", gast.Constant(value="zeros", kind=None)
            ),
            "use_bias": args_kwargs.get("bias", gast.Constant(value=True, kind=None)),
            "feature_group_count": args_kwargs.get(
                "groups", gast.Constant(value=1, kind=None)
            ),
            "input_dilation": args_kwargs.get(
                "dilation", gast.Constant(value=1, kind=None)
            ),
            "kernel_dilation": args_kwargs.get(
                "dilation", gast.Constant(value=1, kind=None)
            ),
        }

        layer_name = "FlaxConv"

        self.create_local_import(node, "FlaxConv")

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
        linear_args = ["in_features", "out_features", "bias"]
        args_kwargs = self.convert_args_to_kwargs(node, linear_args)

        kwargs = {
            "in_features": args_kwargs["in_features"],
            "out_features": args_kwargs["out_features"],
            "use_bias": args_kwargs.get("bias", gast.Constant(value=True, kind=None)),
        }
        self.create_local_import(node, "FlaxLinear")

        return gast.Call(
            func=(
                node.func
                if is_alias
                else gast.Name(id="FlaxLinear", ctx=gast.Load(), annotation=None)
            ),
            args=[],
            keywords=[
                gast.keyword(arg=key, value=value) for key, value in kwargs.items()
            ],
        )

    def convert_batchnorm2d(self, node, is_alias=False):
        batchnorm_args = [
            "num_features",
            "eps",
            "momentum",
            "affine",
            "track_running_stats",
        ]
        args_kwargs = self.convert_args_to_kwargs(node, batchnorm_args)
        axis = 1 if self.data_format == "channels_first" else -1
        kwargs = {
            "num_features": args_kwargs["num_features"],
            "momentum": args_kwargs.get(
                "momentum", gast.Constant(value=0.99, kind=None)
            ),
            "epsilon": args_kwargs.get("eps", gast.Constant(value=1e-5, kind=None)),
            "use_bias": args_kwargs.get("affine", gast.Constant(value=True, kind=None)),
            "use_scale": args_kwargs.get(
                "affine", gast.Constant(value=True, kind=None)
            ),
            "axis": args_kwargs.get("axis", gast.Constant(value=axis, kind=None)),
            "track_running_stats": args_kwargs.get(
                "track_running_stats", gast.Constant(value=True, kind=None)
            ),
        }

        self.create_local_import(node, "FlaxBatchNorm")
        return gast.Call(
            func=(
                node.func
                if is_alias
                else gast.Name(id="FlaxBatchNorm", ctx=gast.Load(), annotation=None)
            ),
            args=[],
            keywords=[
                gast.keyword(arg=key, value=value) for key, value in kwargs.items()
            ],
        )

    def convert_maxpool(self, node, is_alias=False):
        maxpool_args = ["kernel_size", "stride", "padding"]
        args_kwargs = self.convert_args_to_kwargs(node, maxpool_args)
        kwargs = {
            "window_shape": args_kwargs["kernel_size"],
            "strides": args_kwargs.get("stride", gast.Constant(value=1)),
            "padding": gast.Constant(
                value="VALID" if args_kwargs["padding"] == 0 else "SAME", kind=None
            ),
        }

        self.create_local_import(node, "FlaxMaxPool")

        return gast.Call(
            func=(
                node.func
                if is_alias
                else gast.Name(id="FlaxMaxPool", ctx=gast.Load(), annotation=None)
            ),
            args=[],
            keywords=[
                gast.keyword(arg=key, value=value) for key, value in kwargs.items()
            ],
        )

    def get_import_module(self):
        return "jax__stateful_layers"
