import ivy
import torch
import numpy as np
import tensorflow as tf
import os
from packaging.version import parse
import keras

if parse(keras.__version__).major > 2:
    KerasVariable = keras.src.backend.Variable
else:
    KerasVariable = tf.Variable


def get_torch_cnn_model_inputs(model_str: str = "alexnet", target: str = "tensorflow"):
    if model_str == "alexnet":
        n, c, h, w = (128, 3, 128, 128)
        channel_last_perm_dims = [0, 2, 3, 1]
    elif model_str == "unet":
        n, c, h, w = (32, 3, 32, 32)
        channel_last_perm_dims = [0, 2, 3, 1]

    x_np = np.random.randn(n, c, h, w).astype("float32")
    x = torch.from_numpy(x_np)

    if target == "tensorflow":
        # Note that here we permute the dimensions to make the data format
        # channel last for tensorflow which is what the tensorflow translated
        # model expects to run on because of our AST optimizations during translation

        # TODO: remove this comment once transpose optimization is working
        x_np_transposed = x_np  # np.transpose(x_np, channel_last_perm_dims)
        x_translated = ivy.native_array(x_np_transposed)
    else:
        x_translated = ivy.native_array(x_np)

    return x, x_translated


def _retrive_layer(model, key):
    if len(key.split(".")) == 1:
        return model, key

    module_path, weight_name = key.rsplit(".", 1)

    # Retrieve the layer using the module path
    layer = model
    for attr in module_path.split("."):
        layer = getattr(layer, attr)

    return layer, weight_name


def _maybe_update_keras_layer_weights(layer, weight_name, new_weight):
    # Update the weight in the retrieved layer
    if hasattr(layer, weight_name):
        weight_var = getattr(layer, weight_name)
        if isinstance(weight_var, tf.Variable):
            weight_var.assign(tf.Variable(new_weight, dtype=weight_var.dtype))
        elif isinstance(weight_var, KerasVariable):
            weight_var.assign(
                KerasVariable(new_weight, dtype=weight_var.dtype, name=weight_var.name)
            )
        else:
            setattr(
                layer,
                weight_name,
                tf.convert_to_tensor(new_weight, dtype=weight_var.dtype),
            )
    else:
        raise AttributeError(
            f"Layer '{layer}' does not have a weight named '{weight_name}'"
        )


def sync_models_torch(model1, model2):
    """
    Synchronizes the parameters and buffers of the original and the translated model.

    Args:
        model1 (torch.nn.Module): The original PyTorch model.
        model2 (ivy.Module converted tf.keras.Model)): The converted ivy.Module converted tf.keras.Model model.

    Returns:
        None
    """
    has_keras_layers = os.environ.get("USE_NATIVE_KERAS_LAYERS", None) == "true"
    transpose_weights = (
        has_keras_layers
        or os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", None) == "true"
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
            f"Missing params TF model: {missing_in_params2}\n"
        )
    missing_in_buffers2 = buffers1.keys() - buffers2.keys()
    if missing_in_buffers2:
        raise AssertionError(
            f"Mismatch in buffers keys:\n"
            f"Missing buffers in TF model: {missing_in_buffers2}\n"
        )

    # Set the parameters and buffers of the second model to be the same as the first model
    with torch.no_grad():
        for name in params1:
            layer, weight_name = _retrive_layer(model2, key_mapping[name])

            params1_np = params1[name].cpu().detach().numpy()
            # Transpose the parameters to match the TensorFlow format
            if (
                transpose_weights
                and "DepthwiseConv" in layer.__class__.__name__
                and len(params1_np.shape) == 4
            ):  # DepthConvolutional layer
                params1_np = np.transpose(params1_np, (2, 3, 0, 1))
            elif (
                transpose_weights
                and "Conv" in layer.__class__.__name__
                and len(params1_np.shape) == 4
            ):  # Convolutional layer
                params1_np = np.transpose(params1_np, (2, 3, 1, 0))
            elif (
                "Dense" in layer.__class__.__name__
                and len(params1_np.shape) == 2
                and layer.built
            ):  # Dense layer
                params1_np = np.transpose(params1_np, (1, 0))

            # inplace update the native keras layer. This is done as the parameters in
            # self.v are a different copy than the parameters in self.weights. Hence, we
            # need to explicitly update self.weights, otherwise the changes won't reflect.
            if layer.__class__.__name__.startswith("Keras"):
                _maybe_update_keras_layer_weights(
                    layer=layer, weight_name=weight_name, new_weight=params1_np
                )
                params2[name] = getattr(layer, weight_name)
                continue

            params2[name].assign(tf.Variable(params1_np, dtype=params2[name].dtype))

        for name in buffers1:
            layer, weight_name = _retrive_layer(model2, key_mapping[name])

            buffers1_np = buffers1[name].cpu().detach().numpy()
            if (
                transpose_weights
                and "DepthwiseConv" in layer.__class__.__name__
                and len(params1_np.shape) == 4
            ):  # DepthConvolutional layer
                params1_np = np.transpose(params1_np, (2, 3, 0, 1))
            elif (
                transpose_weights
                and "Conv" in layer.__class__.__name__
                and len(params1_np.shape) == 4
            ):  # Convolutional layer
                buffers1_np = np.transpose(buffers1_np, (2, 3, 1, 0))
            elif (
                "Dense" in layer.__class__.__name__
                and len(params1_np.shape) == 2
                and layer.built
            ):  # Dense layer
                buffers1_np = np.transpose(buffers1_np, (1, 0))

            # inplace update the native keras layer. This is done as the parameters in
            # self.v are a different copy than the parameters in self.weights. Hence, we
            # need to explicitly update self.weights, otherwise the changes won't reflect.
            if layer.__class__.__name__.startswith("Keras"):
                _maybe_update_keras_layer_weights(
                    layer=layer, weight_name=weight_name, new_weight=buffers1_np
                )
                buffers2[name] = getattr(layer, weight_name)
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
        if (
            transpose_weights
            and "DepthwiseConv" in layer.__class__.__name__
            and len(params2_np.shape) == 4
        ):  # Convolutional layer
            params2_np = np.transpose(params2_np, (2, 3, 0, 1))
        elif (
            transpose_weights
            and "Conv" in layer.__class__.__name__
            and len(params2_np.shape) == 4
        ):  # Convolutional layer
            params2_np = np.transpose(params2_np, (3, 2, 0, 1))
        elif (
            "Dense" in layer.__class__.__name__
            and len(params1_np.shape) == 2
            and layer.built
        ):  # Dense layer
            params2_np = np.transpose(params2_np, (1, 0))

        assert np.allclose(
            params1_np, params2_np
        ), f"Mismatch found in parameters: {name}"

    for name in buffers1:
        layer, weight_name = _retrive_layer(model2, key_mapping[name])

        buffers1_np = buffers1[name].cpu().detach().numpy()
        buffers2_np = buffers2[name].numpy()

        # Transpose the parameters back to the PyTorch format for comparison
        if (
            transpose_weights
            and "DepthwiseConv" in layer.__class__.__name__
            and len(params2_np.shape) == 4
        ):  # Convolutional layer
            params2_np = np.transpose(params2_np, (2, 3, 0, 1))
        elif (
            transpose_weights
            and "Conv" in layer.__class__.__name__
            and len(params2_np.shape) == 4
        ):  # Convolutional layer
            buffers2_np = np.transpose(buffers2_np, (3, 2, 0, 1))
        elif (
            "Dense" in layer.__class__.__name__
            and len(params1_np.shape) == 2
            and layer.built
        ):  # Dense layer
            buffers2_np = np.transpose(buffers2_np, (1, 0))

        assert np.allclose(
            buffers1_np, buffers2_np
        ), f"Mismatch found in buffers: {name}"
