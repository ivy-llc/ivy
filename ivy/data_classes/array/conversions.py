"""Collection of Ivy functions for wrapping functions to accept and return ivy.Array
instances.
"""

# global
import numpy as np
from typing import Any, Union, Tuple, Dict, Iterable, Optional

# local
import ivy


# Helpers #
# --------#


def _to_native(x: Any, inplace: bool = False) -> Any:
    if isinstance(x, ivy.Array):
        return x.data
    elif isinstance(x, ivy.Shape):
        return x.shape
    elif isinstance(x, ivy.Container):
        return x.cont_map(
            lambda x_, _: _to_native(x_, inplace=inplace), inplace=inplace
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
    include_derived: Optional[Dict[type, bool]] = None,
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
    """Returns the input array converted to an ivy.Array instance if it is a frontend
    array type, otherwise the input is returned unchanged. If nested is set, the check
    is applied to all nested leafs of tuples, lists and dicts contained within x.

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
        return ivy.nested_map(x, _to_ivy, include_derived, shallow=False)
    return _to_ivy(x)


def args_to_ivy(
    *args: Iterable[Any],
    include_derived: Optional[Dict[type, bool]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """Returns args and keyword args in their ivy.Array or form for all nested
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
    native_args = ivy.nested_map(args, _to_ivy, include_derived, shallow=False)
    native_kwargs = ivy.nested_map(kwargs, _to_ivy, include_derived, shallow=False)
    return native_args, native_kwargs


def to_native(
    x: Union[ivy.Array, ivy.NativeArray, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[type, bool]] = None,
    cont_inplace: bool = False,
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
    """Returns the input item in its native backend framework form if it is an
    ivy.Array instance, otherwise the input is returned unchanged. If nested is set,
    the check is applied to all nested leaves of tuples, lists and dicts contained
    within ``x``.

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

    Returns
    -------
     ret
        the input in its native framework form in the case of ivy.Array instances.
    """
    if nested:
        return ivy.nested_map(
            x,
            lambda x: _to_native(x, inplace=cont_inplace),
            include_derived,
            shallow=False,
        )
    return _to_native(x, inplace=cont_inplace)


def args_to_native(
    *args: Iterable[Any],
    include_derived: Dict[type, bool] = None,
    cont_inplace: bool = False,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """Returns args and keyword args in their native backend framework form for all
    nested ivy.Array instances, otherwise the arguments are returned unchanged.

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
    kwargs
        The key-word arguments to check

    Returns
    -------
     ret
        the same arguments, with any nested ivy.Array or instances converted to their
        native form.

    """
    native_args = ivy.nested_map(
        args,
        lambda x: _to_native(x, inplace=cont_inplace),
        include_derived,
        shallow=False,
    )
    native_kwargs = ivy.nested_map(
        kwargs,
        lambda x: _to_native(x, inplace=cont_inplace),
        include_derived,
        shallow=False,
    )
    return native_args, native_kwargs
