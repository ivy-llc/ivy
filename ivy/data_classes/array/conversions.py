"""Ivy wrapping functions for conversions.

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


def _array_to_new_backend(
    x: Union[ivy.Array, ivy.NativeArray], native: bool = False
) -> Union[ivy.Array, ivy.NativeArray]:
    # Frontend instances
    if hasattr(x, "_ivy_array"):
        return x

    # ivy.Array instances
    native_x = x.data if isinstance(x, ivy.Array) else x
    native_x_type = type(native_x).__name__

    # Modify native_type here since @tf.function converts tf.EagerTensor into
    # tf.Tensor when running @tf.function enclosed function
    if ivy.backend == "tensorflow":
        import tensorflow as tf

        native_x_type = (
            "EagerTensor"
            if not tf.executing_eagerly() and isinstance(native_x, tf.Tensor)
            else native_x_type
        )

    if native_x_type not in ARRAY_TO_BACKEND:
        return x

    # Check if the other possible backends match with the native data type
    native_x_backend = ARRAY_TO_BACKEND[native_x_type]

    # Handle the `Tensor` name clash in paddle and torch
    if not isinstance(native_x_backend, str):
        native_x_backend = "torch" if "torch" in str(native_x.__class__) else "paddle"

    # If the current backend and the backend for the given array match,
    # simply return the array as is
    if ivy.backend == native_x_backend:
        if native:
            return native_x
        np_intermediary = ivy.to_numpy(native_x)
        return ivy.array(np_intermediary)

    # Otherwise, convert to the new backend
    else:
        native_x_backend = ivy.with_backend(native_x_backend)
        # Handle native variable instances here
        if native_x_backend.gradients._is_variable(native_x):
            x_data = native_x_backend.gradients._variable_data(native_x)
            # x_data = _array_to_new_backend(x_data, native=True)
            from ivy.functional.ivy.gradients import _variable

            return _variable(x_data).data if native else _variable(x_data)

        np_intermediary = native_x_backend.to_numpy(native_x)
        ret = ivy.array(np_intermediary)
        return ret.data if native else ret


def _to_new_backend(
    x: Any,
    native: bool = False,
    inplace: bool = False,
    to_ignore: tuple = (),
) -> Any:
    if isinstance(x, ivy.Container):
        to_ignore = ivy.default(to_ignore, ())
        return x.cont_map(
            lambda x_, _: _to_new_backend(
                x_, native=native, inplace=inplace, to_ignore=to_ignore
            ),
            inplace=inplace,
        )
    return _array_to_new_backend(x, native=native)


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
    """Return the input array converted to an ivy.Array instance if it is a
    native array type, otherwise the input is returned unchanged. If nested is
    set, the check is applied to all nested leafs of tuples, lists and dicts
    contained within x.

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
    """Return args and keyword args in their ivy.Array or form for all nested
    instances, otherwise the arguments are returned unchanged.

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
    """Return the input item in its native backend framework form if it is an
    ivy.Array instance, otherwise the input is returned unchanged. If nested is
    set, the check is applied to all nested leaves of tuples, lists and dicts
    contained within ``x``.

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
    include_derived: Optional[Dict[str, bool]] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """Return args and keyword args in their native backend framework form for
    all nested ivy.Array instances, otherwise the arguments are returned
    unchanged.

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


def to_new_backend(
    x: Union[ivy.Array, ivy.NativeArray, Iterable],
    native: bool = True,
    nested: bool = False,
    include_derived: Optional[Dict[str, bool]] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
    """Return the input array converted to new backend framework form if it is
    an `ivy.Array`, `ivy.NativeArray` or NativeVariable instance. If nested is
    set, the check is applied to all nested leaves of tuples, lists and dicts
    contained within ``x``.

    Parameters
    ----------
    x
        The input to maybe convert.
    native
        Whether to return the new array as a `ivy.NativeArray`, NativeVariable
        or an `ivy.Array`. Default is ``True``.
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
        the input in the new backend framework form in the case of array instances.
    """
    if nested:
        return ivy.nested_map(
            lambda x: _to_new_backend(
                x, native=native, inplace=cont_inplace, to_ignore=to_ignore
            ),
            x,
            include_derived,
            shallow=False,
        )
    return _to_new_backend(x, native=native, inplace=cont_inplace, to_ignore=to_ignore)


def args_to_new_backend(
    *args: Iterable[Any],
    native: bool = True,
    shallow: bool = True,
    include_derived: Optional[Dict[str, bool]] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """Return args and keyword args in the new current backend framework for
    all nested ivy.Array, ivy.NativeArray or NativeVariable instances.

    Parameters
    ----------
    args
        The positional arguments to check
    native
        Whether to return the new array as a ivy.NativeArray, NativeVariable
        or an ivy.Array. Default is ``True``.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    cont_inplace
        Whether to update containers in place.
        Default is ``False``
    to_ignore
        Types to ignore when deciding whether to go deeper into the nest or not
    shallow
        Whether to inplace update the input nest or not
        Only works if nest is a mutable type. Default is ``True``.
    kwargs
        The key-word arguments to check

    Returns
    -------
    ret
        The same arguments, with any nested array instances converted
        to the new backend.
    """
    new_args = ivy.nested_map(
        lambda x: _to_new_backend(
            x, native=native, inplace=cont_inplace, to_ignore=to_ignore
        ),
        args,
        include_derived,
        shallow=shallow,
    )
    new_kwargs = ivy.nested_map(
        lambda x: _to_new_backend(
            x, native=native, inplace=cont_inplace, to_ignore=to_ignore
        ),
        kwargs,
        include_derived,
        shallow=shallow,
    )
    return new_args, new_kwargs
