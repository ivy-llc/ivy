import tensorflow
import tensorflow as tf

from typing import Callable
from typing import Optional
from typing import Iterable
from typing import Tuple
from collections import UserDict
from typing import Union
from typing import Dict

from .tensorflow__helpers import tensorflow_default
from .tensorflow__helpers import tensorflow_nested_map
from .tensorflow__helpers import tensorflow_set_item


def tensorflow_nested_map(
    fn: Callable,
    x: Union[tensorflow.Tensor, tf.Tensor, Iterable],
    /,
    include_derived: Optional[Union[Dict[str, bool], bool]] = None,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    to_mutable: bool = False,
    _tuple_check_fn: Optional[Callable] = None,
    _list_check_fn: Optional[Callable] = None,
    _dict_check_fn: Optional[Callable] = None,
    shallow: bool = True,
):
    to_ignore = tensorflow_default(to_ignore, ())
    if include_derived is True:
        include_derived = {"tuple": True, "list": True, "dict": True}
    elif not include_derived:
        include_derived = {}
    for t in ("tuple", "list", "dict"):
        if t not in include_derived:
            include_derived = tensorflow_set_item(include_derived, t, False)
    class_instance = type(x)
    if (
        hasattr(x, "is_tracked_proxy")
        and hasattr(class_instance, "__bases__")
        and not set(class_instance.__bases__).intersection(set(to_ignore))
    ):
        to_ignore = to_ignore + (class_instance,)
    tuple_check_fn = tensorflow_default(
        _tuple_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived["tuple"]
        else lambda x_, t_: type(x_) is t_,
    )
    list_check_fn = tensorflow_default(
        _list_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived["list"]
        else lambda x_, t_: type(x_) is t_,
    )
    dict_check_fn = tensorflow_default(
        _dict_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived["dict"]
        else lambda x_, t_: type(x_) is t_,
    )
    if tuple_check_fn(x, tuple) and not isinstance(x, to_ignore):
        ret_list = [
            tensorflow_nested_map(
                fn,
                i,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for i in x
        ]
        if to_mutable:
            return ret_list
        elif hasattr(x, "_fields"):
            return class_instance(**dict(zip(x._fields, ret_list)))
        else:
            return class_instance(ret_list)
    elif list_check_fn(x, list) and not isinstance(x, to_ignore):
        ret_list = [
            tensorflow_nested_map(
                fn,
                i,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for i in x
        ]
        if shallow:
            x = tensorflow_set_item(x, slice(None, None, None), ret_list[:])
            return x
        return class_instance(ret_list)
    elif (dict_check_fn(x, dict) or isinstance(x, UserDict)) and not isinstance(
        x, to_ignore
    ):
        class_instance = type(x)
        ret = {
            k: tensorflow_nested_map(
                fn,
                v,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for k, v in x.items()
        }
        if shallow:
            x.update(ret)
            return x
        return class_instance(ret)
    elif isinstance(x, slice):
        return slice(*tensorflow_nested_map(fn, [x.start, x.stop, x.step]))
    return fn(x)
