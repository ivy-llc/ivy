import pytest
import gast
import os
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import ConfigurationsContainer
from ivy.transpiler.utils.ast_utils import ast_to_source_code
from ivy.transpiler.transformations.transformers.native_layers_transformer.ivy_to_tf_native_layer_transformer import (
    PytorchToKerasLayer,
)
from ivy.transpiler.transformations.transformers.native_layers_transformer.ivy_to_jax_native_layer_transformer import (
    PytorchToFlaxLayer,
)


def transform_function(func, target):
    os.environ["USE_NATIVE_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"

    container = ConfigurationsContainer()
    container.load_configurations(source="torch", target=target)

    object_like = BaseObjectLike.from_object(func)
    root = gast.parse(object_like.source_code)
    configuration = BaseTransformerConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    if target == "tensorflow":
        converter = PytorchToKerasLayer(root, transformer, configuration)
    elif target == "jax":
        converter = PytorchToFlaxLayer(root, transformer, configuration)
    else:
        raise ValueError(f"Unsupported target: {target}")
    converter.transform()

    del os.environ["USE_NATIVE_LAYERS"]
    del os.environ["APPLY_TRANSPOSE_OPTIMIZATION"]
    return ast_to_source_code(root).strip()


# Test for PyTorch to Keras conversion (Conv2D, with additional kwargs)
def test_pytorch_to_keras_conv2d():
    class PytorchModel:
        def __init__(self, tensorflow_Conv2d):
            self.conv1 = tensorflow_Conv2d(
                3, 64, 7, stride=2, padding=3, dilation=2, groups=1, bias=False
            )

    transformed = transform_function(PytorchModel, "tensorflow")
    assert "from .tensorflow__stateful_layers import KerasConv2D" in transformed
    assert "self.conv1 = KerasConv2D" in transformed
    assert "in_channels=3" in transformed
    assert "filters=64" in transformed
    assert "kernel_size=7" in transformed
    assert "strides=2" in transformed
    assert "padding=3" in transformed
    assert "dilation_rate=2" in transformed
    assert "use_bias=False" in transformed


# Test for PyTorch to Keras conversion (depthwise convolution scenario)
def test_pytorch_to_keras_depthwise_conv2d():
    class PytorchModel:
        def __init__(self, tensorflow_Conv2d):
            self.conv_dw = tensorflow_Conv2d(
                3, 64, 3, stride=1, padding=1, groups=3
            )  # groups=3 for depthwise

    transformed = transform_function(PytorchModel, "tensorflow")
    assert (
        "from .tensorflow__stateful_layers import KerasDepthwiseConv2D" in transformed
    )
    assert "self.conv_dw = KerasDepthwiseConv2D" in transformed
    assert "in_channels=3" in transformed
    assert "filters=64" in transformed
    assert "kernel_size=3" in transformed
    assert "strides=1" in transformed
    assert "padding=1" in transformed


# Test for PyTorch to Keras layer alias mapping (layer as parameter)
def test_pytorch_to_keras_layer_as_parameter():
    tensorflow_Conv2d = lambda *args, **kwargs: None

    class PytorchModel:
        def __init__(self, layer=tensorflow_Conv2d):
            self.custom_layer = layer(3, 64, 7, stride=2, padding=3)

    transformed = transform_function(PytorchModel, "tensorflow")
    assert "from .tensorflow__stateful_layers import KerasConv2D" not in transformed
    assert "self.custom_layer = layer" in transformed  # Ensure alias mapping works
    assert "in_channels=3" in transformed
    assert "filters=64" in transformed
    assert "kernel_size=7" in transformed


# Test for PyTorch to Keras layer alias mapping (layer as local variable)
def test_pytorch_to_keras_layer_as_variable():
    tensorflow_Conv2d = lambda *args, **kwargs: None

    class PytorchModel:
        def __init__(
            self,
        ):
            layer = tensorflow_Conv2d
            self.custom_layer = layer(3, 64, 7, stride=2, padding=3)

    transformed = transform_function(PytorchModel, "tensorflow")
    assert "from .tensorflow__stateful_layers import KerasConv2D" in transformed
    assert "self.custom_layer = layer" in transformed
    assert "in_channels=3" in transformed
    assert "filters=64" in transformed
    assert "kernel_size=7" in transformed


# Test for PyTorch to Flax conversion (Conv2D, with additional kwargs)
def test_pytorch_to_flax_conv2d():
    class PytorchModel:
        def __init__(self, jax_Conv2d):
            self.conv1 = jax_Conv2d(
                3, 64, 7, stride=2, padding=3, dilation=2, groups=1, bias=False
            )

    transformed = transform_function(PytorchModel, "jax")
    assert "from .jax__stateful_layers import FlaxConv" in transformed
    assert "self.conv1 = FlaxConv" in transformed
    assert "in_features=3" in transformed
    assert "out_features=64" in transformed
    assert "kernel_size=7" in transformed
    assert "strides=2" in transformed
    assert "padding=3" in transformed
    assert "padding_mode='zeros'" in transformed
    assert "input_dilation=2" in transformed
    assert "kernel_dilation=2" in transformed
    assert "feature_group_count=1" in transformed
    assert "use_bias=False" in transformed


# Test for PyTorch to Flax conversion depthwise convolution scenario)
def test_pytorch_to_flax_depthwise_conv2d():
    class PytorchModel:
        def __init__(self, jax_Conv2d):
            self.conv_dw = jax_Conv2d(
                3, 64, 3, stride=1, padding=1, groups=3
            )  # groups=3 for depthwise

    transformed = transform_function(PytorchModel, "jax")
    assert "from .jax__stateful_layers import FlaxConv" in transformed
    assert "self.conv_dw = FlaxConv" in transformed
    assert "in_features=3" in transformed
    assert "out_features=64" in transformed
    assert "kernel_size=3" in transformed
    assert "strides=1" in transformed
    assert "padding=1" in transformed
    assert "padding_mode='zeros'" in transformed
    assert "input_dilation=1" in transformed
    assert "kernel_dilation=1" in transformed
    assert "feature_group_count=3" in transformed


# Test for PyTorch to Flax layer alias mapping (layer as parameter)
def test_pytorch_to_flax_layer_as_parameter():
    jax_Conv2d = lambda *args, **kwargs: None

    class PytorchModel:
        def __init__(self, layer=jax_Conv2d):
            self.custom_layer = layer(3, 64, 7, stride=2, padding=3)

    transformed = transform_function(PytorchModel, "jax")
    assert (
        "from .jax__stateful_layers import FlaxConv" not in transformed
    )  # import should be at module scope.
    assert "self.custom_layer = layer" in transformed
    assert "in_features=3" in transformed
    assert "out_features=64" in transformed
    assert "kernel_size=7" in transformed
    assert "strides=2" in transformed
    assert "padding=3" in transformed
    assert "padding_mode='zeros'" in transformed
    assert "input_dilation=1" in transformed
    assert "kernel_dilation=1" in transformed
    assert "feature_group_count=1" in transformed


# Test for PyTorch to Flax layer alias mapping (layer as local variable)
def test_pytorch_to_flax_layer_as_variable():
    jax_Conv2d = lambda *args, **kwargs: None

    class PytorchModel:
        def __init__(
            self,
        ):
            layer = jax_Conv2d
            self.custom_layer = layer(3, 64, 7, stride=2, padding=3)

    transformed = transform_function(PytorchModel, "jax")
    assert (
        "from .jax__stateful_layers import FlaxConv" in transformed
    )  # import should be at module scope.
    assert "self.custom_layer = layer" in transformed
    assert "in_features=3" in transformed
    assert "out_features=64" in transformed
    assert "kernel_size=7" in transformed
    assert "strides=2" in transformed
    assert "padding=3" in transformed
    assert "padding_mode='zeros'" in transformed
    assert "input_dilation=1" in transformed
    assert "kernel_dilation=1" in transformed
    assert "feature_group_count=1" in transformed


# Test for PyTorch to Keras conversion (BatchNorm2D layer)
def test_pytorch_to_keras_batchnorm2d():
    class PytorchModel:
        def __init__(self, tensorflow_BatchNorm2d):
            self.bn = tensorflow_BatchNorm2d(64, eps=1e-05, momentum=0.1)

    transformed = transform_function(PytorchModel, "tensorflow")
    assert "from .tensorflow__stateful_layers import KerasBatchNorm2D" in transformed
    assert "self.bn = KerasBatchNorm2D" in transformed
    assert "num_features=64" in transformed
    assert "epsilon=1e-05" in transformed
    assert "momentum=0.1" in transformed
    assert "center=True" in transformed
    assert "scale=True" in transformed


# Test for PyTorch to Flax conversion (BatchNorm2D layer)
def test_pytorch_to_flax_batchnorm2d():
    class PytorchModel:
        def __init__(self, jax_BatchNorm2d):
            self.bn = jax_BatchNorm2d(64, eps=1e-05, momentum=0.1)

    transformed = transform_function(PytorchModel, "jax")
    assert "from .jax__stateful_layers import FlaxBatchNorm" in transformed
    assert "self.bn = FlaxBatchNorm" in transformed
    assert "num_features=64" in transformed
    assert "epsilon=1e-05" in transformed
    assert "momentum=0.1" in transformed
    assert "use_bias=True" in transformed
    assert "use_scale=True" in transformed


# Test for PyTorch to Keras conversion (Linear layer)
def test_pytorch_to_keras_linear():
    class PytorchModel:
        def __init__(self, tensorflow_Linear):
            self.fc = tensorflow_Linear(512, 1000, bias=False)

    transformed = transform_function(PytorchModel, "tensorflow")
    assert "from .tensorflow__stateful_layers import KerasDense" in transformed
    assert "self.fc = KerasDense" in transformed
    assert "in_features=512" in transformed
    assert "units=1000" in transformed
    assert "use_bias=False" in transformed


# Test for PyTorch to Flax conversion (Linear layer)
def test_pytorch_to_flax_linear():
    class PytorchModel:
        def __init__(self, jax_Linear):
            self.fc = jax_Linear(512, 1000, bias=False)

    transformed = transform_function(PytorchModel, "jax")
    assert "from .jax__stateful_layers import FlaxLinear" in transformed
    assert "self.fc = FlaxLinear" in transformed
    assert "in_features=512" in transformed
    assert "out_features=1000" in transformed
    assert "use_bias=False" in transformed


if __name__ == "__main__":
    pytest.main([__file__])
