"""
Ivy wrapping functions for conversions.

Collection of Ivy functions for wrapping functions to accept and return
ivy.Array instances.
"""

# global
import numpy as np
from typing import Any, Union, Tuple, Dict, Iterable, Optional

# local
import ivy


# Helpers #
# --------#


# TODO: Need to find a better way to do this.
# Temporarily adding as is for the
# `ivy.Module.to_keras_module`method
# for the KLA demo. Do not move/remove.
ARRAY_TO_BACKEND = {
    "ndarray": "numpy",
    "Tensor": ["torch", "paddle"],
    "Parameter": "torch",
    "EagerTensor": "tensorflow",
    "ResourceVariable": "tensorflow",
    "DeviceArray": "jax",
    "Array": "jax",
    "ArrayImpl": "jax",
    "EagerParamBase": "paddle",
}


def _to_native(x: Any, inplace: bool = False, to_ignore: tuple = ()) -> Any:
    to_ignore = ivy.default(to_ignore, ())
    if isinstance(x, to_ignore):
        return x
    if isinstance(x, ivy.Array):
        return x.data
    # to prevent the graph from breaking for the time being
    elif type(x) is ivy.Shape:
        return x.shape
    elif isinstance(x, ivy.Container):
        return x.cont_map(
            lambda x_, _: _to_native(x_, inplace=inplace, to_ignore=to_ignore),
            inplace=inplace,
        )
    return x


def _to_ivy(x: Any) -> Any:
    if isinstance(x, ivy.Array):
        return x
    elif isinstance(x, ivy.NativeShape):
        return ivy.Shape(x)
    elif isinstance(x, ivy.Container):
        return x.to_ivy()
    if ivy.is_native_array(x) or isinstance(x, np.ndarray):
        return ivy.Array(x)
    return x


# Wrapped #
# --------#


def to_ivy(
    x: Union[ivy.Array, ivy.NativeArray, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[str, bool]] = None,
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
    """
    Return the input array converted to an ivy.Array instance if it is a native array
    type, otherwise the input is returned unchanged. If nested is set, the check is
    applied to all nested leafs of tuples, lists and dicts contained within x.

    Parameters
    ----------
    x
        The input to be converted.
    nested
        Whether to apply the conversion on arguments in a nested manner. If so, all
        dicts, lists and tuples will be traversed to their lowest leaves in search of
        ivy.Array instances. Default is ``False``.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict. Default
        is False.

    Returns
    -------
    ret
        the input in its native framework form in the case of ivy.Array or instances.
    """
    if nested:
        return ivy.nested_map(_to_ivy, x, include_derived, shallow=False)
    return _to_ivy(x)


def args_to_ivy(
    *args: Iterable[Any],
    include_derived: Optional[Dict[str, bool]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """
    Return args and keyword args in their ivy.Array or form for all nested instances,
    otherwise the arguments are returned unchanged.

    Parameters
    ----------
    args
        The positional arguments to check
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    kwargs
        The key-word arguments to check

    Returns
    -------
     ret
        the same arguments, with any nested arrays converted to ivy.Array or
        instances.
    """
    native_args = ivy.nested_map(_to_ivy, args, include_derived, shallow=False)
    native_kwargs = ivy.nested_map(_to_ivy, kwargs, include_derived, shallow=False)
    return native_args, native_kwargs


def to_native(
    x: Union[ivy.Array, ivy.NativeArray, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[str, bool]] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
    """
    Return the input item in its native backend framework form if it is an ivy.Array
    instance, otherwise the input is returned unchanged. If nested is set, the check is
    applied to all nested leaves of tuples, lists and dicts contained within ``x``.

    Parameters
    ----------
    x
        The input to maybe convert.
    nested
        Whether to apply the conversion on arguments in a nested manner. If so, all
        dicts, lists and tuples will be traversed to their lowest leaves in search of
        ivy.Array instances. Default is ``False``.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    cont_inplace
        Whether to update containers in place. Default is ``False``
    to_ignore
        Types to ignore when deciding whether to go deeper into the nest or not

    Returns
    -------
     ret
        the input in its native framework form in the case of ivy.Array instances.
    """
    if nested:
        return ivy.nested_map(
            lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
            x,
            include_derived,
            shallow=False,
        )
    return _to_native(x, inplace=cont_inplace, to_ignore=to_ignore)


def args_to_native(
    *args: Iterable[Any],
    include_derived: Dict[str, bool] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """
    Return args and keyword args in their native backend framework form for all nested
    ivy.Array instances, otherwise the arguments are returned unchanged.

    Parameters
    ----------
    args
        The positional arguments to check
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    cont_inplace
        Whether to update containers in place.
        Default is ``False``
    to_ignore
        Types to ignore when deciding whether to go deeper into the nest or not
    kwargs
        The key-word arguments to check

    Returns
    -------
     ret
        the same arguments, with any nested ivy.Array or instances converted to their
        native form.
    """
    native_args = ivy.nested_map(
        lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
        args,
        include_derived,
        shallow=False,
    )
    native_kwargs = ivy.nested_map(
        lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
        kwargs,
        include_derived,
        shallow=False,
    )
    return native_args, native_kwargs


# TODO: Need to find a better way to do this.
# Temporarily adding as is for the
# `ivy.Module.to_keras_module`method
# for the . Do not move/remove.
def array_to_new_backend(
    x: Union[ivy.Array, ivy.NativeArray],
    native: bool = False,
) -> Union[ivy.Array, ivy.NativeArray]:
    native_x = x.data if isinstance(x, ivy.Array) else x
    native_x_type = type(native_x).__name__

    # Modify native_type here since @tf.function converts tf.EagerTensor into
    # tf.Tensor when running @tf.function on a transpiled graph
    if ivy.current_backend_str() == "tensorflow":
        import importlib

        native_x_type = (
            "EagerTensor"
            if not importlib.import_module("tensorflow").executing_eagerly()
            and isinstance(native_x, importlib.import_module("tensorflow").Tensor)
            else native_x_type
        )

    # Check for paddle first, as it shares the 'Tensor' native_x_type with torch
    if "paddle" in str(native_x.__class__) and ivy.current_backend_str() == "paddle":
        if native:
            return native_x
        else:
            return x

    if hasattr(x, "_ivy_array"):
        return x

    # Check if the other possible backends match with the native data type
    if (
        native_x_type in ARRAY_TO_BACKEND
        and ivy.current_backend_str() in ARRAY_TO_BACKEND[native_x_type]
    ):
        if ivy.current_backend_str() == "torch":
            if "torch" in str(native_x.__class__):
                # torch and paddle both use 'Tensor', return if this is torch
                return x
            else:
                # if it's actually a paddle tensor, convert to an ivy array
                ret = ivy.array(native_x.numpy())
                return ret.data if native else ret
        if ivy.current_backend_str() == "paddle":
            if "paddle" in str(native_x.__class__):
                # torch and paddle both use 'Tensor', return if this is paddle
                return x
            else:
                # if it's actually a torch tensor, convert to an ivy array
                ret = ivy.array(native_x.numpy())
                return ret.data if native else ret
        return x

    if native_x_type not in ARRAY_TO_BACKEND:
        return x
    native_x = (
        native_x.detach().cpu()
        if native_x_type in ["Parameter", "Tensor"]
        else native_x
    )
    np_intermediary = np.array(native_x)
    ret = ivy.array(np_intermediary)
    return ret.data if native else ret


# TODO: Need to find a better way to do this.
# Temporarily adding as is for the
# `ivy.Module.to_keras_module()`method
# for the KLA demo. Do not move/remove.
def nest_array_to_new_backend(
    nest: Iterable[Union[ivy.Array, ivy.NativeArray]],
    native: bool = True,
    shallow: bool = True,
) -> Iterable[Union[ivy.Array, ivy.NativeArray]]:
    """
    Convert a given ivy.Array or ivy.NativeArray to a new backend framework.

    Parameters
    ----------
    nest
        Input nest with the leaves to be converted to a new backend.
    native
        Whether to return the new array as a ivy.NativeArray or an ivy.Array.
        Default is ``True``.
    shallow
        Whether to inplace update the input nest or not
        Only works if nest is a mutable type. Default is ``True``.

    Returns
    -------
    ret
        The input nest with leaves converted to the new backend framework.
    """
    return ivy.nested_map(
        lambda x: array_to_new_backend(x, native=native),
        nest,
        include_derived=True,
        shallow=shallow,
    )
