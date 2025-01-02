"""Holds helper methods for all sorts of conversions that are used across the tracer-transpiler repository"""

import numpy as np
import copy
import enum
from typing import Union, Tuple, Iterable
from collections import UserDict

import ivy
from .numpy_proxy import (
    NewNDArray,
    NUMPY_TO_CUSTOM,
    CUSTOM_TO_NUMPY,
)
from .tracked_var_proxy import (
    TrackedVarProxy,
    type_to_proxy,
    get_types_to_ignore,
    PROXY_TO_BUILTIN_TYPES,
    PROXY_ITERATOR_TO_TYPES,
)


# Checks #
# ------ #


def is_frontend_array(x) -> bool:
    return hasattr(x, "ivy_array")


def is_frontend_shape(x) -> bool:
    return hasattr(x, "ivy_shape")


def is_array(x, with_numpy: bool = False):
    is_from_numpy = isinstance(x, NewNDArray) or (
        isinstance(x, ivy.Array) and isinstance(x.data, NewNDArray)
    )
    return ivy.is_array(x) or is_frontend_array(x) or (is_from_numpy and with_numpy)


# Native arrays #
# ------------- #


def _to_native(x, inplace: bool = False, to_ignore: tuple = None):
    to_ignore = ivy.default(to_ignore, ())
    if isinstance(x, to_ignore):
        return x
    if isinstance(x, ivy.Array):
        return x.data
    elif isinstance(x, ivy.Shape):
        return x.shape
    elif is_frontend_array(x):
        return x.ivy_array.data
    elif is_frontend_shape(x):
        return x.ivy_shape.shape
    elif isinstance(x, ivy.Container):
        return x.cont_map(
            lambda x_, _: _to_native(x_, inplace=inplace), inplace=inplace
        )
    return x


def to_native(x, cont_inplace: bool = False, to_ignore: tuple = None):
    """Converts the input x to equivalent native framework data structures in a nested manner"""
    return ivy.nested_map(
        lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
        x,
        shallow=False,
    )


def frontend_arrays_to_ivy(x):
    return x.ivy_array if hasattr(x, "ivy_array") else x


def _batched_tracer_to_array(obj):
    if hasattr(obj, "batch_dim"):
        return obj.val
    return obj


def remove_batch_dim(args, kwargs):
    _to_ignore = get_types_to_ignore()
    args = ivy.nested_map(
        lambda x: x[0] if is_array(x) else x,
        args,
        to_ignore=_to_ignore,
        shallow=False,
    )
    kwargs = ivy.nested_map(
        lambda x: x[0] if is_array(x) else x,
        kwargs,
        to_ignore=_to_ignore,
        shallow=False,
    )
    return args, kwargs


# Numpy proxies #
# ------------- #


def _to_ND(x):
    """Converts numpy ndarrays/scalars to instances of our custom subclasses
    ``NewNDArray``, ``NewFloat64`` etc. It does this even if the ndarray is contained
    within an ivy or frontend array.
    """
    if type(x) in NUMPY_TO_CUSTOM:
        return NUMPY_TO_CUSTOM[type(x)](x)
    elif isinstance(x, ivy.Array) and type(x.data) in NUMPY_TO_CUSTOM:
        return ivy.Array(NUMPY_TO_CUSTOM[type(x.data)](x.data))
    elif is_frontend_array(x) and type(x.ivy_array.data) in NUMPY_TO_CUSTOM:
        x.ivy_array.data = NUMPY_TO_CUSTOM[type(x.ivy_array.data)](x.ivy_array.data)
        return x
    return x


def _from_ND(x):
    """Converts instances of our custom subclasses ``NewNDArray``,
    ``NewFloat64`` etc. to numpy ndarrays/scalars. It does this even if
    the NewNDArray is contained within an ivy or frontend array.
    """
    if type(x) in CUSTOM_TO_NUMPY:
        return x.data
    elif isinstance(x, ivy.Array) and type(x.data) in CUSTOM_TO_NUMPY:
        return ivy.Array(CUSTOM_TO_NUMPY[type(x.data)](x.data))
    elif is_frontend_array(x) and type(x.ivy_array.data) in CUSTOM_TO_NUMPY:
        x.ivy_array.data = CUSTOM_TO_NUMPY[type(x.ivy_array.data)](x.ivy_array.data)
        return x
    return x


def to_custom_numpy_type(x, with_numpy):
    """
    This is used to convert any ``numpy.ndarray`` or scalar input arguments
    to the graph to our custom numpy types. It is also used in the op logging
    wrapped function to ensure that all numpy creation functions
    (``linspace``, ``random.uniform``,...) return custom types.

    We do this to ensure that during op logging all the parameters in
    the graph are our custom classes, which we need to do in order to be
    able to track instance methods (necessary because ``numpy.ndarray``
    methods can't be inplace updated with our wrapped function, whereas
    the subclasses' methods can be).
    """
    if ivy.current_backend_str() == "numpy" or with_numpy:
        return ivy.nested_map(_to_ND, x, include_derived={"dict": True})
    return x


def _custom_to_numpy(args, kwargs):
    """Converts our custom subclass of numpy ndarrays
    back to numpy ndarrays/scalars.
    """
    _to_ignore = get_types_to_ignore()
    args = ivy.nested_map(
        lambda x: np.array(x) if isinstance(x, NewNDArray) else x,
        args,
        shallow=False,
        to_ignore=_to_ignore,
    )
    kwargs = ivy.nested_map(
        lambda x: np.array(x) if isinstance(x, NewNDArray) else x,
        kwargs,
        shallow=False,
        to_ignore=_to_ignore,
    )
    return args, kwargs


# Between backends #
# ---------------- #

# ToDo: map ResourceVariable's to our own frontend variable
NATIVE_ARRAY_TO_FRONTEND = {}

ARRAY_TO_BACKEND = {
    "ndarray": "numpy",
    "NewNDArray": "numpy",
    "Tensor": ["torch", "paddle"],
    "Parameter": "torch",
    "EagerTensor": "tensorflow",
    "ResourceVariable": "tensorflow",
    "Variable": "tensorflow",
    "DeviceArray": "jax",
    "Array": "jax",
    "ArrayImpl": "jax",
    "EagerParamBase": "paddle",
}

BACKENDS = ("tensorflow", "jax", "torch", "paddle", "numpy")

imported_frontends = []


def _tf_to_numpy(x):
    "Memory efficient conversion to numpy from tf tensors."
    try:
        return x.read_value()._numpy()
    except:
        return x.numpy()


def _get_ivy_device(x):
    if hasattr(x, "_ivy_array"):
        x = x._ivy_array
    if isinstance(x, ivy.Array):
        x = x.data
    if isinstance(x, NewNDArray):
        x = x._data
    for backend in BACKENDS:
        if backend in x.__class__.__module__:
            ivy_backend = ivy.with_backend(backend)
            break
    return ivy_backend.dev(x)


def native_array_to_frontend(x: Union[ivy.Array, ivy.NativeArray]):
    """Converts a native/ivy array into the corresponding ivy frontend
    array. Also the array contained within the frontend array will be
    from the globally set ivy backend.
    """
    x = x.data if isinstance(x, ivy.Array) else x

    if "numpy" not in imported_frontends:
        import ivy.functional.frontends.numpy as np_frontend

        NATIVE_ARRAY_TO_FRONTEND["ndarray"] = np_frontend.array
        imported_frontends.append("numpy")

    if "jax" in str(x.__class__) and "jax" not in imported_frontends:
        import ivy.functional.frontends.jax as jax_frontend

        NATIVE_ARRAY_TO_FRONTEND["DeviceArray"] = jax_frontend.numpy.array
        NATIVE_ARRAY_TO_FRONTEND["Array"] = jax_frontend.numpy.array
        NATIVE_ARRAY_TO_FRONTEND["ArrayImpl"] = jax_frontend.numpy.array
        imported_frontends.append("jax")

    elif "torch" in str(x.__class__) and "torch" not in imported_frontends:
        import ivy.functional.frontends.torch as torch_frontend

        NATIVE_ARRAY_TO_FRONTEND["torch.Tensor"] = torch_frontend.tensor
        NATIVE_ARRAY_TO_FRONTEND["Parameter"] = (
            torch_frontend.tensor
        )  # TODO: change to torch_frontend.Parameter
        imported_frontends.append("torch")

    elif "tensorflow" in str(x.__class__) and "tensorflow" not in imported_frontends:
        import ivy.functional.frontends.tensorflow as tf_frontend

        NATIVE_ARRAY_TO_FRONTEND["EagerTensor"] = tf_frontend.constant
        NATIVE_ARRAY_TO_FRONTEND["ResourceVariable"] = (
            tf_frontend.python.ops.resource_variable_ops.ResourceVariable
        )
        NATIVE_ARRAY_TO_FRONTEND["Variable"] = tf_frontend.Variable
        imported_frontends.append("tensorflow")

    elif "paddle" in str(x.__class__) and "paddle" not in imported_frontends:
        import ivy.functional.frontends.paddle as paddle_frontend

        NATIVE_ARRAY_TO_FRONTEND["EagerParamBase"] = paddle_frontend.to_tensor
        NATIVE_ARRAY_TO_FRONTEND["paddle.Tensor"] = paddle_frontend.to_tensor
        imported_frontends.append("paddle")

    if is_frontend_array(x):
        return x
    x_type = type(x).__name__
    if x_type in ARRAY_TO_BACKEND:
        x_device = _get_ivy_device(x)
        if x_type == "Tensor":  # resolves torch & paddle "Tensor" conflict
            source = x.__class__.__module__
            mod_type = source + ".Tensor"
        else:
            source = ARRAY_TO_BACKEND[x_type]
            mod_type = x_type
        if not ivy.current_backend_str() == "tensorflow":
            # Tensorflow has a problem with buffers
            # with respect to memory alignment
            try:
                stride = []
                if hasattr(x, "strides"):
                    stride = x.strides
                elif hasattr(x, "stride"):
                    stride = x.stride()
                if len(stride) > 1 and any(
                    s1 < s2 for s1, s2 in zip(stride, stride[1:])
                ):
                    raise NotImplementedError  # TODO bug in JAX framework dlpack
                capsule = ivy.with_backend(source).to_dlpack(x)
                x_ivy = ivy.to_device(ivy.from_dlpack(capsule), x_device)
                return NATIVE_ARRAY_TO_FRONTEND[mod_type](x_ivy)
            except Exception as e:
                pass

        try:
            x = (
                x.detach().cpu().numpy()
                if x_type in ["Parameter", "Tensor", "EagerParamBase"]
                else x
            )
            x = _tf_to_numpy(x) if x_type in ["EagerTensor", "ResourceVariable"] else x
            x_ivy = ivy.array(np.asarray(x), device=x_device)
            return NATIVE_ARRAY_TO_FRONTEND[mod_type](x_ivy)
        except Exception as e:
            pass

    return x


def native_array_to_new_frontend(x, frontend_type):
    """Converts an array to a new frontend array.

    The array passed is usually a native array, but can also be an ivy or frontend array.
    The frontend type can be any of the supported ivy frontend array types.
    """
    # If an instance of ivy.Array
    x = x.data if isinstance(x, ivy.Array) else x

    # If an instance of frontend arrays
    if is_frontend_array(x):
        if type(x) is frontend_type:
            return x
        else:
            return frontend_type(x._ivy_array)

    # If an instance of native arrays
    x_type = type(x).__name__
    try:
        x_device = _get_ivy_device(x)
        x = x.detach().cpu() if x_type in ["Parameter", "Tensor"] else x
        if frontend_type.__module__ == "ivy.functional.frontends.torch.tensor":
            return frontend_type(
                ivy.array(np.array(x), device=x_device), _init_overload=True
            )
        return frontend_type(ivy.array(np.array(x), device=x_device))
    except:
        return x


def nest_native_array_to_new_frontend(nest, input_nest):
    """Take native arrays (might reside in a different framework)
    in nest to new frontend based on the frontend arrays
    contained within the input_nest."""
    _to_ignore = get_types_to_ignore()
    input_nest_frontend_idxs = ivy.nested_argwhere(
        input_nest, is_frontend_array, to_ignore=_to_ignore
    )
    input_nest_types = ivy.map_nest_at_indices(
        input_nest, input_nest_frontend_idxs, type
    )
    if input_nest_types:
        frontend_type = input_nest_types[0]
        nest = ivy.nested_map(
            lambda x: native_array_to_new_frontend(x, frontend_type=frontend_type),
            nest,
            to_ignore=_to_ignore,
        )
        return nest, True
    return nest, False


def array_to_new_backend(
    x: Union[ivy.Array, ivy.NativeArray],
    native: bool = False,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Converts a given Ivy array or Native array to a new backend framework.

    Parameters
    ----------
    x
        Array to be converted to a new backend
    native
        Whether to return the new array as a native array (True) or ivy array (False)
    with_numpy
        Whether numpy arrays can be considered part of the new backend

    Returns
    -------
    Ivy array or native array in the new backend framework
    """
    native_x = x.data if isinstance(x, ivy.Array) else x
    native_x_type = type(native_x).__name__

    # Modify native_type here since @tf.function converts tf.EagerTensor into
    # tf.Tensor when running @tf.function on a transpiled graph
    if ivy.current_backend_str() == "tensorflow":
        import tensorflow as tf

        native_x_type = (
            "EagerTensor"
            if not tf.executing_eagerly() and isinstance(native_x, tf.Tensor)
            else native_x_type
        )

    # Check for paddle first, as it shares the 'Tensor' native_x_type with torch
    if "paddle" in str(native_x.__class__) and ivy.current_backend_str() == "paddle":
        if native:
            return native_x
        else:
            return x

    if is_frontend_array(x):
        return x

    # Check if the other possible backends match with the native data type
    if (
        native_x_type in ARRAY_TO_BACKEND
        and ivy.current_backend_str() in ARRAY_TO_BACKEND[native_x_type]
    ):
        x_device = _get_ivy_device(x)
        if ivy.current_backend_str() == "torch":
            if "torch" in str(native_x.__class__):
                # torch and paddle both use 'Tensor', return if this is torch
                return x
            else:
                # if it's actually a paddle tensor, convert to an ivy array
                ret = ivy.array(native_x.cpu().numpy(), device=x_device)
                return ret.data if native else ret
        if ivy.current_backend_str() == "paddle":
            if "paddle" in str(native_x.__class__):
                # torch and paddle both use 'Tensor', return if this is paddle
                return x
            else:
                # if it's actually a torch tensor, convert to an ivy array
                ret = ivy.array(native_x.cpu().numpy(), device=x_device)
                return ret.data if native else ret
        return x

    if native_x_type not in ARRAY_TO_BACKEND:
        return x
    x_device = _get_ivy_device(x)
    native_x = (
        native_x.detach().cpu()
        if native_x_type in ["Parameter", "Tensor"]
        else native_x
    )
    np_intermediary = np.array(native_x)
    ret = ivy.array(np_intermediary, device=x_device)
    return ret.data if native else ret


def nest_array_to_new_backend(
    nest, native=True, to_ignore=None, shallow=True
):
    return ivy.nested_map(
        lambda x: array_to_new_backend(x, native=native),
        nest,
        include_derived=True,
        to_ignore=to_ignore,
        shallow=shallow,
    )


# Dtypes #
# ------ #


def _to_ivy_dtype(dtype):
    if ivy.is_native_array(dtype):
        return dtype
    if isinstance(dtype, (int, float, complex, bool)):
        return dtype
    if ivy.is_native_dtype(dtype) or any(
        dtype is t for t in (bool, float, int, complex)
    ):
        return ivy.as_ivy_dtype(dtype)
    return dtype


def _to_ivy_device(device):
    # need try statement as `NativeDevice` is just `str` for most
    # backends, and so non device `str`s will enter `as_ivy_dev` often.
    # ToDO: change when we have better checks for native devices
    if not isinstance(device, ivy.NativeDevice) or ivy.current_backend_str() == "numpy":
        return device
    try:
        return ivy.as_ivy_dev(device)
    except:
        return device


def _dtype_and_dev_to_ivy(x):
    _temp = _to_ivy_device(x)
    _temp = _to_ivy_dtype(_temp)
    return _temp


def _convert_to_ivy_dtype(args, kwargs, to_ivy: bool):
    """Converts all args and kwargs to ivy dtypes when transpiling to ivy"""
    if to_ivy:
        _to_ignore = get_types_to_ignore()
        args = ivy.nested_map(
            lambda x: ivy.as_ivy_dtype(x) if isinstance(x, ivy.NativeDtype) else x,
            args,
            shallow=False,
            to_ignore=_to_ignore,
        )
        kwargs = ivy.nested_map(
            lambda x: ivy.as_ivy_dtype(x) if isinstance(x, ivy.NativeDtype) else x,
            kwargs,
            shallow=False,
            to_ignore=_to_ignore,
        )
    return args, kwargs


# Tracked Variable Proxy #
# ---------------------- #


# ToDo: Dynamic control flow
def track(
    var: Union[int, float, bool, str, list, tuple, dict, Iterable],
    with_numpy: bool = True,
    stateful_classes: Tuple = (),
    _deepcopy=True,
) -> TrackedVarProxy:
    """Recursively wraps an arbitrary variable or an iterable of abitrary variables
    in order to track its usage and effect on a traced function.
    Note_1: While the value of the variable will be tracked, the dynamic control flow of the traced
    function will not be re-evaluated.
    Note_2: When wrapping bool variables, the not keyword will break tracking.

    Parameters
    ----------
    var
        Variable to track.
    with_numpy
        Whether we are tracing the graph with numpy
    stateful_classes
        Classes to be considered stateful during tracing
    _deepcopy
        Whether to perform deepcopy of var before tracking

    Returns
    -------
    Derived class of TrackedVarProxy that mirrors the behaviour of var and will be tracked during tracing.
    """
    from tracer.helpers import _is_tracked_variable, _is_untrackable

    if _is_tracked_variable(var):
        return var

    ret = None
    _cls = type(var)
    if ivy.exists(var) and isinstance(var, (list, tuple)):
        # Make sure to track only those aribitrary vars which themselves don't
        # contain other wrapped or stateful classes we'll already be logging
        ret = [
            (
                track(
                    v,
                    with_numpy=with_numpy,
                    stateful_classes=stateful_classes,
                    _deepcopy=_deepcopy,
                )
                if not _is_untrackable(
                    v, with_numpy=with_numpy, stateful_classes=stateful_classes
                )
                else v
            )
            for v in var
        ]

    if ivy.exists(var) and isinstance(var, (dict, UserDict)):
        ret = dict(
            (
                track(
                    (k, v),
                    with_numpy=with_numpy,
                    stateful_classes=stateful_classes,
                    _deepcopy=_deepcopy,
                )
                if not _is_untrackable(
                    (k, v), with_numpy=with_numpy, stateful_classes=stateful_classes
                )
                else (k, v)
            )
            for k, v in var.items()
        )

    if ivy.exists(ret) and not _is_untrackable(
        ret, with_numpy=with_numpy, stateful_classes=stateful_classes
    ):
        return _track(ret, _cls=_cls, _deepcopy=_deepcopy)
    elif ret:
        return _cls(ret)

    return _track(var, _deepcopy=_deepcopy)


def _track(
    var: Union[int, float, bool, str, list, tuple, dict],
    _cls=None,
    _deepcopy: bool = True,
) -> TrackedVarProxy:
    """Converts an arbitrary variable into its TrackedVarProxy counterpart.

    For example, a tuple will be converted into a TrackedTupleProxy.

    Parameters
    ----------
    var
        Variable to track
    _cls: Union[type, tf.TensorShape, torch.Size, ivy.Shape]
        The __class__ of the original variable to be tracked
        (not necessarily the same as that of the var passed into this function)
    _deepcopy
        Whether to perform deepcopy of var before tracking

    Returns
    -------
    Derived class of TrackedVarProxy that mirrors the behaviour of var and will be tracked during tracing
    """
    var = _cls(var) if _cls else var
    type_str = type(var).__name__

    # Need to track enums since they are subclasses of int but their type_str is not enum.Enum
    if isinstance(var, enum.IntEnum):
        type_str = "IntEnum"
    elif isinstance(var, enum.Enum):
        type_str = "Enum"

    # Retreive the type_to_proxy dict
    _type_to_proxy = type_to_proxy()

    if type_str in _type_to_proxy:
        if _deepcopy:
            var = copy.deepcopy(var)
        return _type_to_proxy[type_str](var)
    else:
        return var
        # ToDo: Raise warning? Maybe do this with an isinstance check


def untrack(var: Union[TrackedVarProxy, Iterable[TrackedVarProxy]]):
    """
    Recursively untracks tracked variables or iterable of tracked variables.
    Parameters
    ----------
    var
        Variable or an iterable of variables to recursively untrack.

    Returns
    -------
    Untracked var or iterable of vars
    """
    from tracer.helpers import _is_tracked_variable

    _cls = type(var)
    _typ = None

    # If input is a TrackedVarProxy class
    if _cls in PROXY_TO_BUILTIN_TYPES:
        _typ = PROXY_TO_BUILTIN_TYPES[_cls]

    # elif input is a TrackedVarIteratorProxy class
    elif _cls in PROXY_ITERATOR_TO_TYPES:
        _typ = (
            PROXY_ITERATOR_TO_TYPES[_cls] if _cls in PROXY_ITERATOR_TO_TYPES else _typ
        )
        _typ = "ivy." + _typ if _typ and _typ == "Shape" else _typ

    cls = eval(_typ) if _typ else _cls

    var = var.get_var() if _is_tracked_variable(var) else var

    if isinstance(var, (list, tuple)):
        if type(var).__name__ not in PROXY_TO_BUILTIN_TYPES.values():
            return var
        ret = [untrack(v) for v in var]
        return (
            cls(ret)
            if not hasattr(var, "_fields")
            else cls(**dict(zip(var._fields, ret)))
        )

    if isinstance(var, (dict, UserDict)):
        if type(var).__name__ not in PROXY_TO_BUILTIN_TYPES.values():
            return var
        ret = [untrack((k, v)) for (k, v) in var.items()]
        return cls(ret)

    return var
