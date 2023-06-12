# global
import ivy

# local
import ivy.functional.backends.jax.__init__
import ivy.functional.backends.mxnet.__init__
import ivy.functional.backends.numpy.__init__
import ivy.functional.backends.paddle.__init__
import ivy.functional.backends.tensorflow.__init__
import ivy.functional.backends.torch.__init__


def is_valid_device(device: str, /):
    if device not in ivy.current_backend().__init__.valid_devices:
        if device in ivy.current_backend().__init__.invalid_devices:
            return False, "{} is an invalid device".format(device)
        return False, "{} is not a device".format(device)
    return True, ""


"""
import ivy.functional.backends.jax.__init__ as jx
import ivy.functional.backends.mxnet.__init__ as mxn
import ivy.functional.backends.numpy.__init__ as np
import ivy.functional.backends.paddle.__init__ as pdl
import ivy.functional.backends.tensorflow.__init__ as tf
import ivy.functional.backends.torch.__init__ as torch


def jax_valid_device(device: str, /):
    if device not in jx.valid_devices:
        if device in jx.invalid_devices:
            return False, "{} is an invalid device for jax".format(device)
        return False, "{} is not a device".format(device)
    return True, ""


def mxnet_valid_device(device: str, /):
    if device not in mxn.valid_devices:
        if device in mxn.invalid_devices:
            return False, "{} is an invalid device for mxnet".format(device)
        return False, "{} is not a device".format(device)
    return True, ""


def numpy_valid_device(device: str, /):
    if device not in np.valid_devices:
        if device in np.invalid_devices:
            return False, "{} is an invalid device for numpy".format(device)
        return False, "{} is not a device".format(device)
    return True, ""


def paddle_valid_device(device: str, /):
    if device not in pdl.valid_devices:
        if device in pdl.invalid_devices:
            return False, "{} is an invalid device for paddle".format(device)
        return False, "{} is not a device".format(device)
    return True, ""


def tensorflow_valid_device(device: str, /):
    if device not in tf.valid_devices:
        if device in tf.invalid_devices:
            return False, "{} is an invalid device for tensorflow".format(device)
        return False, "{} is not a device".format(device)
    return True, ""


def torch_valid_device(device: str, /):
    if device not in torch.valid_devices:
        if device in torch.invalid_devices:
            return False, "{} is an invalid device for pytorch".format(device)
        return False, "{} is not a device".format(device)
    return True, ""
"""
