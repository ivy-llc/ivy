import inspect
import functools
from typing import Callable

from .tensorflow__helpers import tensorflow__check_in_nested_sequence
from .tensorflow__helpers import tensorflow__get_preferred_device
from .tensorflow__helpers import tensorflow_asarray
from .tensorflow__helpers import tensorflow_get_item
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_set_item


def tensorflow_handle_array_like_without_promotion(fn: Callable):
    @functools.wraps(fn)
    def _handle_array_like_without_promotion(*args, **kwargs):
        args = list(args)
        num_args = len(args)
        try:
            type_hints = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        parameters = list(type_hints.keys())
        annotations = [param.annotation for param in type_hints.values()]
        device = tensorflow__get_preferred_device(args, kwargs)
        for i, (annotation, parameter, arg) in enumerate(
            zip(annotations, parameters, args)
        ):
            annotation_str = str(annotation)
            if (
                ("rray" in annotation_str or "Tensor" in annotation_str)
                and parameter != "out"
                and all(
                    sq not in annotation_str
                    for sq in ["Sequence", "List", "Tuple", "float", "int", "bool"]
                )
            ):
                if i < num_args:
                    if tensorflow__check_in_nested_sequence(
                        arg, value=Ellipsis, _type=slice
                    ):
                        continue
                    if not tensorflow_is_array(arg):
                        args = tensorflow_set_item(
                            args, i, tensorflow_asarray(arg, device=device)
                        )
                elif parameters in kwargs:
                    kwarg = tensorflow_get_item(kwargs, parameter)
                    if not tensorflow_is_array(kwarg):
                        kwargs = tensorflow_set_item(
                            kwargs, parameter, tensorflow_asarray(kwarg, device=device)
                        )
        return fn(*args, **kwargs)

    _handle_array_like_without_promotion.handle_array_like_without_promotion = True
    return _handle_array_like_without_promotion
