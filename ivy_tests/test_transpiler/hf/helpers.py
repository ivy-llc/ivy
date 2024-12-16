import jax
import tensorflow as tf
from ivy_tests.test_transpiler.translations.helpers import sync_models_torch


def _backend_compile(obj, target):
    if target == "tensorflow":
        return tf.function(obj)
    elif target == "jax":
        return jax.jit(obj)
    return obj


def _target_to_simplified(target: str):
    """
    Convert the name of a target framework to its simplified form,
    such as 'tensorflow' -> 'tf'.
    """
    if target == "numpy":
        return "np"
    if target == "tensorflow":
        return "tf"
    if target == "jax":
        return "jax"
    if target == "torch":
        return "pt"
    return target


def _compute_module_dict_tf(model, prefix=""):
    _module_dict = dict()
    for key, value in model.__dict__.items():
        if isinstance(value, (tf.keras.Model, tf.keras.layers.Layer)):
            if not hasattr(value, "named_parameters"):
                _module_dict.update(_compute_module_dict_tf(value, prefix=f"{key}."))
            else:
                _module_dict[prefix + key] = value
    return _module_dict


def _compute_module_dict_pt(model, keychains):
    _module_dict = dict()
    for keychain in keychains:
        keys = keychain.split(".")
        value = model
        for key in keys:
            value = getattr(value, key)
        _module_dict[keychain] = value
    return _module_dict


def sync_models_HF_torch_to_tf(model_pt, model_tf):

    all_submods_tf = _compute_module_dict_tf(model_tf)
    all_submods_pt = _compute_module_dict_pt(
        model_pt, keychains=list(all_submods_tf.keys())
    )

    for pt_model, tf_model in zip(all_submods_pt.values(), all_submods_tf.values()):
        pt_model.eval()
        tf_model.eval()
        sync_models_torch(pt_model, tf_model)
