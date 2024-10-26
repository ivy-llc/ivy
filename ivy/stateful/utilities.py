import os
from typing import Union, TYPE_CHECKING
from packaging.version import parse
import numpy as np
import ivy

if TYPE_CHECKING:
    from ivy.functional.backends.tensorflow import Model as KerasModel
    from ivy.functional.backends.jax import Model as FlaxModel
    import torch.nn as nn
    import keras
    import flax.nnx as nnx


def _is_submodule(obj, kw):
    cls_str = {
        "torch": ("torch.nn.modules.module.Module",),
        "keras": (
            "keras.engine.training.Model",
            "tf_keras.src.engine.training.Model",
            "keras.src.models.model.Model",
            "tf_keras.src.engine.base_layer.Layer",
            "keras.src.engine.base_layer.Layer",
            "keras.src.layers.layer.Layer",
        ),
        "flax": (
            "flax.nnx.module.Module",
            "flax.nnx.nnx.module.Module",
            "transformers.modeling_flax_utils.FlaxPreTrainedModel",
        ),
    }[kw]
    try:
        for bc in type(obj).mro():
            if any(cls in str(bc) for cls in cls_str):
                return True
    except TypeError:
        pass
    return False


def _compute_module_dict_pt(model, keychains):
    _module_dict = dict()
    for keychain in keychains:
        keys = keychain.split(".")
        value = model
        for key in keys:
            value = getattr(value, key)
        _module_dict[keychain] = value
    return _module_dict


def _retrive_layer(model, key):
    if len(key.split(".")) == 1:
        return model, key

    module_path, weight_name = key.rsplit(".", 1)

    # Retrieve the layer using the module path
    layer = model
    for attr in module_path.split("."):
        layer = getattr(layer, attr)

    return layer, weight_name

def transpose_weights_pt_to_tf_jax(layer, params_np, transpose_weights, fw):
    """
    Transpose weights from PyTorch to TensorFlow/JAX format.

    Args:
    - layer: The layer object.
    - params_np: The weights in NumPy format.
    - transpose_weights: Flag to enable weight transposition.

    Returns:
    - Transposed weights.
    """
    if not transpose_weights:
        return params_np

    if fw == 'jax' and "ConvTranspose" in layer.__class__.__name__ and len(params_np.shape) == 4:
        return np.transpose(params_np, (2, 3, 0, 1))
    elif fw == 'tensorflow' and "DepthwiseConv" in layer.__class__.__name__ and len(params_np.shape) == 4:  
        # Depthwise Convolutional layer
        return np.transpose(params_np, (2, 3, 0, 1))
    elif "Conv" in layer.__class__.__name__:
        if len(params_np.shape) == 3:  # 1D Convolutional layer
            return np.transpose(params_np, (2, 1, 0))
        elif len(params_np.shape) == 4:  # 2D Convolutional layer
            return np.transpose(params_np, (2, 3, 1, 0))
        elif len(params_np.shape) == 5:  # 3D Convolutional layer
            return np.transpose(params_np, (2, 3, 4, 1, 0))
    elif fw == 'jax' and "Linear" in layer.__class__.__name__ and len(params_np.shape) == 2:
        return np.transpose(params_np, (1, 0))
    elif fw == 'tensorflow' and "Dense" in layer.__class__.__name__ and len(params_np.shape) == 2:
        return np.transpose(params_np, (1, 0))

    return params_np


def transpose_weights_tf_jax_to_pt(layer, params_np, transpose_weights, fw):
    """
    Transpose weights from TensorFlow/JAX to PyTorch format.

    Args:
    - layer: The layer object.
    - params_np: The weights in NumPy format.
    - transpose_weights: Flag to enable weight transposition.

    Returns:
    - Transposed weights.
    """
    if not transpose_weights:
        return params_np

    if fw == 'jax' and "ConvTranspose" in layer.__class__.__name__ and len(params_np.shape) == 4:
        return np.transpose(params_np, (2, 3, 0, 1))
    elif fw == 'tensorflow' and  "DepthwiseConv" in layer.__class__.__name__ and len(params_np.shape) == 4:  
        # Depthwise Convolutional layer
        return  np.transpose(params_np, (2, 3, 0, 1))
    elif "Conv" in layer.__class__.__name__:
        if len(params_np.shape) == 3:  # 1D Convolutional layer
            return np.transpose(params_np, (2, 1, 0))
        elif len(params_np.shape) == 4:  # 2D Convolutional layer
            return np.transpose(params_np, (3, 2, 0, 1))
        elif len(params_np.shape) == 5:  # 3D Convolutional layer
            return np.transpose(params_np, (4, 3, 2, 1, 0))
    elif fw == 'jax' and "Linear" in layer.__class__.__name__ and len(params_np.shape) == 2:
        return np.transpose(params_np, (1, 0))
    elif fw == 'tensorflow' and "Dense" in layer.__class__.__name__ and len(params_np.shape) == 2:
        return np.transpose(params_np, (1, 0))
   
    return params_np
    
def _sync_models_torch_and_jax(model1: "nn.Module", model2: "FlaxModel"):
    """Synchronizes the parameters and buffers of the original and the
    translated model.

    Args:
    ----
        model1 (torch.nn.Module): The original PyTorch model.
        model2 (ivy.Module converted Flax.nnx.Module)): The converted ivy.Module converted Flax.nnx.Module.

    Returns:
    -------
        None
    """

    def _pt_name_to_flax_name(layer, weight_name):
        if layer.__class__.__name__ in ("FlaxConv", "FlaxLinear"):
            param_and_buff_map = {
                "weight": "kernel",
                "bias": "bias",
            }
        elif layer.__class__.__name__ == "FlaxBatchNorm":
            param_and_buff_map = {
                "weight": "scale",
                "bias": "bias",
                "running_mean": "mean",
                "running_var": "var",
                "num_batches_tracked": "num_batches_tracked",
            }
        else:
            raise ValueError(f"Layer '{layer}' is not supported.")

        return param_and_buff_map[weight_name]

    def _maybe_update_flax_layer_weights(
        layer, weight_name, new_weight, original_weight
    ):
        # Update the weight in the retrieved layer
        if hasattr(layer, weight_name):
            layer._built = True
            weight_var = getattr(layer, weight_name)
            if isinstance(weight_var, nnx.Variable):
                weight_var.value = jnp.asarray(new_weight, dtype=weight_var.value.dtype)
            else:
                setattr(
                    layer,
                    weight_name,
                    jnp.asarray(new_weight, dtype=weight_var.dtype),
                )

            # now also update the PT placeholder weights for this layer
            layer._built = False
            pt_weight_name = (
                "pt_weight"
                if weight_name == "weight"
                else "pt_bias" if weight_name == "bias" else weight_name
            )
            setattr(
                layer,
                pt_weight_name,
                jnp.asarray(original_weight, dtype=weight_var.dtype),
            )
        else:
            raise AttributeError(
                f"Layer '{layer}' does not have a weight named '{weight_name}'"
            )

    import torch
    import flax.nnx as nnx
    import jax.numpy as jnp

    has_flax_layers = os.environ.get("USE_NATIVE_FW_LAYERS", "true") == "true"
    transpose_weights = (
        has_flax_layers
        or os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", "true") == "true"
    )

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    buffers1 = dict(model1.named_buffers())
    buffers2 = dict(model2.named_buffers())
    # TODO: remove this once the stateful attribute name-conflict has been resolved.
    key_mapping = {}
    for k in params2.keys():
        key_mapping[k.replace("pt_", "")] = k

    for k in buffers2.keys():
        key_mapping[k.replace("pt_", "")] = k

    params2 = {k.replace("pt_", ""): v for k, v in params2.items()}
    buffers2 = {k.replace("pt_", ""): v for k, v in buffers2.items()}

    # Check if both models have the same parameters and buffers
    missing_in_params2 = params1.keys() - params2.keys()
    if missing_in_params2:
        raise AssertionError(
            f"Mismatch in param keys:\n"
            f"Missing params Flax model: {missing_in_params2}\n"
        )
    missing_in_buffers2 = buffers1.keys() - buffers2.keys()
    if missing_in_buffers2:
        raise AssertionError(
            f"Mismatch in buffers keys:\n"
            f"Missing buffers in Flax model: {missing_in_buffers2}\n"
        )

    # Set the parameters and buffers of the second model to be the same as the first model
    with torch.no_grad():
        for name in params1:
            layer, weight_name = _retrive_layer(model2, key_mapping[name])

            params1_np = params1[name].cpu().detach().numpy()
            # Transpose the parameters to match the JAX format
            params1_np = transpose_weights_pt_to_tf_jax(layer, params1_np, transpose_weights,fw='jax')
            # inplace update the native Flax layer. This is done as the parameters in
            # self.v are a different copy than the layer's original parameters. Hence, we
            # need to explicitly update the layer's original parameters, otherwise the changes won't reflect.
            if layer.__class__.__name__.startswith("Flax"):
                flax_name = _pt_name_to_flax_name(layer, weight_name)
                _maybe_update_flax_layer_weights(
                    layer=layer,
                    weight_name=weight_name,
                    new_weight=params1_np,
                    original_weight=params1[name].cpu().detach().numpy(),
                )
                params2[name] = getattr(layer, flax_name)
                continue

            if isinstance(params2[name], nnx.Variable):
                params2[name].value = jnp.asarray(
                    params1_np, dtype=params2[name].value.dtype
                )
            else:
                params2[name] = jnp.asarray(params1_np, dtype=params2[name].dtype)
                setattr(model2, name, params2[name])

        for name in buffers1:
            layer, weight_name = _retrive_layer(model2, key_mapping[name])

            buffers1_np = buffers1[name].cpu().detach().numpy()
            # Transpose the buffers to match the JAX format
            buffers1_np = transpose_weights_pt_to_tf_jax(layer, buffers1_np, transpose_weights,fw='jax')

            # inplace update the native Flax layer. This is done as the buffers in
            # self.buffers are a different copy than the layer's original buffers. Hence, we
            # need to explicitly update the layer's original buffers, otherwise the changes won't reflect.
            if layer.__class__.__name__.startswith("Flax"):
                flax_name = _pt_name_to_flax_name(layer, weight_name)
                _maybe_update_flax_layer_weights(
                    layer=layer,
                    weight_name=weight_name,
                    new_weight=buffers1_np,
                    original_weight=buffers1[name].cpu().detach().numpy(),
                )
                buffers2[name] = getattr(layer, flax_name)
                continue

            if isinstance(buffers2[name], nnx.Variable):
                buffers2[name].value = jnp.asarray(
                    buffers1_np, dtype=buffers2[name].value.dtype
                )

            else:
                buffers2[name] = jnp.asarray(buffers1_np, dtype=buffers2[name].dtype)
                setattr(model2, name, buffers2[name])

    # Check if the parameters and buffers are the same
    for name in params1:
        layer, weight_name = _retrive_layer(model2, key_mapping[name])

        params1_np = params1[name].cpu().detach().numpy()
        params2_np = (
            params2[name].value._value
            if isinstance(params2[name], nnx.Variable)
            else params2[name]._value
        )
        # Transpose the parameters back to the PyTorch format for comparison
        params2_np = transpose_weights_tf_jax_to_pt(layer, params2_np, transpose_weights,fw='jax')

        assert np.allclose(
            params1_np, params2_np
        ), f"Mismatch found in parameters: {name}"

    for name in buffers1:
        layer, weight_name = _retrive_layer(model2, key_mapping[name])

        buffers1_np = buffers1[name].cpu().detach().numpy()
        buffers2_np = (
            buffers2[name].value._value
            if isinstance(buffers2[name], nnx.Variable)
            else buffers2[name]._value
        )

        # Transpose the buffers back to the PyTorch format for comparison
        buffers2_np = transpose_weights_tf_jax_to_pt(layer, buffers2_np, transpose_weights,fw='jax')

        assert np.allclose(
            buffers1_np, buffers2_np
        ), f"Mismatch found in buffers: {name}"


def _sync_models_torch_and_tf(model1: "nn.Module", model2: "KerasModel"):
    """Synchronizes the parameters and buffers of the original and the
    translated model.

    Args:
    ----
        model1 (torch.nn.Module): The original PyTorch model.
        model2 (ivy.Module converted keras.Model)): The converted ivy.Module converted keras.Model.

    Returns:
    -------
        None
    """
    def _pt_name_to_keras_name(layer, weight_name):
        if layer.__class__.__name__ in ("KerasConv2D", "KerasDense"):
            param_and_buff_map = {
                "weight": "_kernel",
                "bias": "bias",
            }
        elif layer.__class__.__name__ == "KerasDepthwiseConv2D":
            if parse(keras.__version__).major > 2:
                param_and_buff_map = {
                    "weight": "kernel",
                    "bias": "bias",
                }
            else:
                param_and_buff_map = {
                    "weight": "depthwise_kernel",
                    "bias": "bias",
                }
        elif layer.__class__.__name__ == "KerasBatchNorm2D":
            param_and_buff_map = {
                "weight": "gamma",
                "bias": "beta",
                "running_mean": "moving_mean",
                "running_var": "moving_variance",
                "num_batches_tracked": "num_batches_tracked",
            }
        else:
            raise ValueError(f"Layer '{layer}' is not supported.")

        return param_and_buff_map[weight_name] 
    
    def _maybe_update_keras_layer_weights(layer, weight_name, new_weight, original_weight):
        # Update the weight in the retrieved layer
        if hasattr(layer, weight_name):
            layer._is_built = True
            weight_var = getattr(layer, weight_name)
            if isinstance(weight_var, tf.Variable):
                weight_var.assign(tf.Variable(new_weight, dtype=weight_var.dtype))
            elif isinstance(weight_var, KerasVariable):
                weight_var.assign(
                    KerasVariable(
                        new_weight, dtype=weight_var.dtype, name=weight_var.name
                    )
                )
            else:
                setattr(
                    layer,
                    weight_name,
                    tf.convert_to_tensor(original_weight, dtype=weight_var.dtype),
                )
            # now also update the PT placeholder weights for this layer
            layer._is_built = False
            pt_weight_name = (
                "pt_weight"
                if weight_name == "weight"
                else "pt_bias" if weight_name == "bias" else weight_name
            )
            setattr(
                layer,
                pt_weight_name,
                None if original_weight is None else tf.convert_to_tensor(original_weight, dtype=weight_var.dtype),
            )
        else:
            raise AttributeError(
                f"Layer '{layer}' does not have a weight named '{weight_name}'"
            )

    import torch
    import tensorflow as tf
    import keras

    if parse(keras.__version__).major > 2:
        KerasVariable = keras.src.backend.Variable
    else:
        KerasVariable = tf.Variable

    has_keras_layers = os.environ.get("USE_NATIVE_FW_LAYERS", "true") == "true"
    transpose_weights = (
        has_keras_layers
        or os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", "true") == "true"
    )

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    buffers1 = dict(model1.named_buffers())
    buffers2 = dict(model2.named_buffers())
    # TODO: remove this once the stateful attribute name-conflict has been resolved.
    key_mapping = {}
    for k in params2.keys():
        key_mapping[k.replace("pt_", "")] = k

    for k in buffers2.keys():
        key_mapping[k.replace("pt_", "")] = k

    params2 = {k.replace("pt_", ""): v for k, v in params2.items()}
    buffers2 = {k.replace("pt_", ""): v for k, v in buffers2.items()}

    # Check if both models have the same parameters and buffers
    assert params1.keys() == params2.keys()
    assert buffers1.keys() == buffers2.keys()

    # Set the parameters and buffers of the second model to be the same as the first model
    with torch.no_grad():
        for name in params1:
            layer, weight_name = _retrive_layer(model2, key_mapping[name])

            params1_np = params1[name].cpu().detach().numpy()
            # Transpose the parameters to match the TensorFlow format
            params1_np = transpose_weights_pt_to_tf_jax(layer, params1_np, transpose_weights, fw='tensorflow')

            # inplace update the native keras layer. This is done as the parameters in
            # self.v are a different copy than the parameters in self.weights. Hence, we
            # need to explicitly update self.weights, otherwise the changes won't reflect.
            if layer.__class__.__name__.startswith("Keras"):
                keras_name = _pt_name_to_keras_name(layer, weight_name)
                _maybe_update_keras_layer_weights(
                    layer=layer, weight_name=weight_name, new_weight=params1_np, original_weight=params1[name].cpu().detach().numpy()
                )
                params2[name] = getattr(layer, keras_name)
                continue

            params2[name].assign(tf.Variable(params1_np, dtype=params2[name].dtype))

        for name in buffers1:
            layer, weight_name = _retrive_layer(model2, key_mapping[name])

            buffers1_np = buffers1[name].cpu().detach().numpy()
            # Transpose the buffers to match the TensorFlow format
            buffers1_np = transpose_weights_pt_to_tf_jax(layer, buffers1_np, transpose_weights, fw='tensorflow')

            # inplace update the native keras layer. This is done as the parameters in
            # self.v are a different copy than the parameters in self.weights. Hence, we
            # need to explicitly update self.weights, otherwise the changes won't reflect.
            if layer.__class__.__name__.startswith("Keras"):
                keras_name = _pt_name_to_keras_name(layer, weight_name)
                _maybe_update_keras_layer_weights(
                    layer=layer, weight_name=weight_name, new_weight=buffers1_np,original_weight=buffers1[name].cpu().detach().numpy()
                )
                buffers2[name] = getattr(layer, keras_name)
                continue

            if isinstance(buffers2[name], tf.Variable):
                buffers2[name].assign(
                    tf.Variable(buffers1_np, dtype=buffers2[name].dtype)
                )
            else:
                buffers2[name] = tf.convert_to_tensor(
                    buffers1_np, dtype=buffers2[name].dtype
                )

    # Check if the parameters and buffers are the same
    for name in params1:
        layer, weight_name = _retrive_layer(model2, key_mapping[name])

        params1_np = params1[name].cpu().detach().numpy()
        params2_np = params2[name].numpy()
        # Transpose the parameters back to the PyTorch format for comparison
        params2_np = transpose_weights_tf_jax_to_pt(layer, params2_np, transpose_weights, fw='tensorflow')

        assert np.allclose(
            params1_np, params2_np
        ), f"Mismatch found in parameters: {name}"

    for name in buffers1:
        layer, weight_name = _retrive_layer(model2, key_mapping[name])

        buffers1_np = buffers1[name].cpu().detach().numpy()
        buffers2_np = buffers2[name].numpy()
        # Transpose the buffers back to the PyTorch format for comparison
        buffers2_np = transpose_weights_tf_jax_to_pt(layer, buffers2_np, transpose_weights, fw='tensorflow')

        assert np.allclose(
            buffers1_np, buffers2_np
        ), f"Mismatch found in buffers: {name}"


def sync_models_torch_and_tf(
    model_pt: "nn.Module", model_tf: Union["keras.Model", "KerasModel"]
):
    """Synchronizes the weights and buffers between a PyTorch model
    (`torch.nn.Module`) and a TensorFlow model (`keras.Model`).

    This function ensures that both models have identical parameters and buffers by
    iterating through their submodules and synchronizing them. The TensorFlow model
    must either be an instance of `KerasModel` or have submodules that inherit from the
    translated `KerasModel`/`KerasLayer`, and expose interfaces similar to `torch.nn.Module`,
    including `named_parameters()` and `named_buffers()`.

    Args:
    ----
        model_pt (torch.nn.Module): The PyTorch model to synchronize from.
        model_tf (keras.Model): The TensorFlow model to synchronize to, with submodules
                                inheriting from the custom `KerasModel`/`KerasLayer` class.

    Returns:
    -------
        None


    Example:
    -------
        ```python
        import torch.nn as nn
        import keras

        #`CustomKerasLinear` is a subclass of `Layer` that exposes a similar
        # interface to torch.nn.Module (with named_parameters and named_buffers).
        class CustomKerasLinear(Layer):
            def __init__(self, in_features, out_features):
                super(CustomKerasLinear, self).__init__()
                self.weight = tf.Variable(tf.random.normal([out_features, in_features]))
                self.bias = tf.Variable(tf.random.normal([out_features]))

            def call(self, x):
                return tf.matmul(x, self.weight) + self.bias

            def named_parameters(self):
                        return [("weight", self.weight), ("bias", self.bias)]

            def named_buffers(self):
                        return []

            def eval(self):
                return False

        #`NativeKerasModel` is a subclass of keras.Model and does NOT exposes a similar
        # interface to torch.nn.Module (with named_parameters and named_buffers).
        class NativeKerasModel(keras.Model):
            def __init__(self):
                super(NativeKerasModel, self).__init__()
                self.linear = CustomKerasLinear(10, 5)

            def call(self, x):
                return self.linear(x)

        class PyTorchModel(nn.Module):
            def __init__(self):
                super(PyTorchModel, self).__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        # Instantiate both models
        model_pt = PyTorchModel()  # PyTorch model
        model_tf = NativeKerasModel()  # Native Keras model inheriting from keras.Model

        # Sync all submodules between the PyTorch and Keras models
        sync_models_torch_and_tf(model_pt, model_tf)
        ```
    """

    def _compute_module_dict_tf(model, prefix=""):
        _module_dict = dict()
        for key, value in model.__dict__.items():
            if isinstance(value, (tf.keras.Model, tf.keras.layers.Layer)):
                if not hasattr(value, "named_parameters"):
                    _module_dict.update(
                        _compute_module_dict_tf(value, prefix=f"{key}.")
                    )
                else:
                    _module_dict[prefix + key] = value
        return _module_dict

    try:
        pass
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`torch` was not found installed on your system. Please proceed "
            "to install it and restart your interpreter to see the changes."
        ) from exc

    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`tensorflow` was not found installed on your system. Please proceed "
            "to install it and restart your interpreter to see the changes."
        ) from exc

    if hasattr(model_tf, "named_parameters"):
        _sync_models_torch_and_tf(model_pt, model_tf)
    else:
        all_submods_tf = _compute_module_dict_tf(model_tf)
        all_submods_pt = _compute_module_dict_pt(
            model_pt, keychains=list(all_submods_tf.keys())
        )

        for pt_model, tf_model in zip(all_submods_pt.values(), all_submods_tf.values()):
            pt_model.eval()
            tf_model.eval()
            _sync_models_torch_and_tf(pt_model, tf_model)


def sync_models_torch_and_jax(
    model_pt: "nn.Module", model_jax: Union["nnx.Module", "FlaxModel"]
):
    """Synchronizes the weights and buffers between a PyTorch model
    (`torch.nn.Module`) and a Flax model (`flax.nnx.Module`).

    This function ensures both models have identical parameters and buffers by
    iterating through their submodules and synchronizing them. The Flax model must
    either be an instance of `FlaxModel` or have submodules that inherit from the
    translated `FlaxModel`, and expose interfaces similar to `torch.nn.Module`,
    including `named_parameters()` and `named_buffers()`.

    Args:
    ----
        model_pt (torch.nn.Module): The PyTorch model to synchronize from.
        model_flax (flax.nnx.Module): The Flax model to synchronize to, with submodules
                                      inheriting from the custom `FlaxModel` class.

    Returns:
    -------
        None

    Example:
    -------
        ```python
        import torch.nn as nn
        import jax.numpy as jnp
        import flax.nnx as nnx

        #`CustomFlaxLinear` is a subclass of `FlaxModel` that exposes a similar
        # interface to torch.nn.Module (with named_parameters and named_buffers).
        class CustomFlaxLinear(FlaxModel):
            def __init__(self, in_features, out_features):
                super(CustomFlaxLinear, self).__init__()
                self.weight = nnx.Param(jax.random.normal(jax.random.key(0), [out_features,in_features]))
                self.bias = nnx.Param(jax.random.normal(jax.random.key(0),[out_features]))

            def call(self, x):
                return x @ self.weight + bias

            def named_parameters(self):
                        return [("weight", self.weight), ("bias", self.bias)]

            def named_buffers(self):
                        return []

            def eval(self):
                return False

        #`NativeFlaxModel` is a subclass of nnx.Module and does NOT exposes a similar
        # interface to torch.nn.Module (with named_parameters and named_buffers).
        class NativeFlaxModel(nnx.Module):
            def __init__(self):
                super(NativeFlaxModel, self).__init__()
                self.linear = CustomFlaxLinear(10, 5)

            def call(self, x):
                return self.linear(x)

        class PyTorchModel(nn.Module):
            def __init__(self):
                super(PyTorchModel, self).__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        # Instantiate both models
        model_pt = PyTorchModel()  # PyTorch model
        model_flax = NativeFlaxModel()  # Native Flax model inheriting from nnx.Module

        # Sync all submodules between the PyTorch and Keras models
        sync_models_torch_and_jax(model_pt, model_flax)
        ```
    """

    def _compute_module_dict_jax(model, prefix=""):
        _module_dict = dict()
        for key, value in model.__dict__.items():
            if isinstance(value, nnx.Module) and value != model:
                if not hasattr(value, "named_parameters"):
                    _module_dict.update(
                        _compute_module_dict_jax(value, prefix=f"{key}.")
                    )
                else:
                    _module_dict[prefix + key] = value
        return _module_dict

    try:
        import torch  # noqa
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`torch` was not found installed on your system. Please proceed "
            "to install it and restart your interpreter to see the changes."
        ) from exc

    try:
        import flax  # noqa

        version = parse(flax.__version__)
        if version < parse("0.8.0"):
            raise ImportError(
                "Flax version 0.8.0 or higher is required. Please update your Flax installation."
            )
        import flax.nnx as nnx  # noqa
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`flax` was not found installed on your system. Please proceed "
            "to install it and restart your interpreter to see the changes."
        ) from exc

    try:
        import jax  # noqa
        import jax.numpy as jnp  # noqa
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`jax` was not found installed on your system. Please proceed "
            "to install it and restart your interpreter to see the changes."
        ) from exc

    if hasattr(model_jax, "named_parameters"):
        _sync_models_torch_and_jax(model_pt, model_jax)

    else:
        all_submods_jax = _compute_module_dict_jax(model_jax)
        all_submods_pt = _compute_module_dict_pt(
            model_pt, keychains=list(all_submods_jax.keys())
        )

        for pt_model, jax_model in zip(
            all_submods_pt.values(), all_submods_jax.values()
        ):
            pt_model.eval()
            jax_model.eval()
            _sync_models_torch_and_jax(pt_model, jax_model)


def sync_models(
    original_model: "nn.Module",
    translated_model: Union["keras.Model", "KerasModel", "nnx.Module", "FlaxModel"],
):
    """Synchronizes the weights and buffers between a native PyTorch model
    (`torch.nn.Module`) and it's translated version in TensorFlow or Flax.

    Args:
    ----
        original_model (torch.nn.Module): The PyTorch model to synchronize from.
        translated_model (tf.keras.Model or nnx.Module): The target model to synchronize to,
                                                  either a TensorFlow or Flax model.
    """
    if not _is_submodule(original_model, "torch"):
        raise ivy.utils.exceptions.IvyException(
            "sync_models expected an instance of `nn.Module` as the first argument. got {}".format(
                original_model
            )
        )
    if _is_submodule(translated_model, "keras"):
        sync_models_torch_and_tf(original_model, translated_model)

    elif _is_submodule(translated_model, "flax"):
        sync_models_torch_and_jax(original_model, translated_model)
    else:
        raise ivy.utils.exceptions.IvyNotImplementedException(
            "sync_models expected an instance of a `keras.Model` or `nnx.Module` as the second argument. got {}".format(
                translated_model
            )
        )

    print("All parameters and buffers are now synced!")
