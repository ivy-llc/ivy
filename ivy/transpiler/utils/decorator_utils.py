import inspect
import types
from ..exceptions import exceptions
from ivy.utils.decorator_utils import (
    handle_get_item,
    handle_set_item,
    handle_methods,
    dummy_inplace_update,
    handle_transpose_in_input_and_output,
    handle_transpose_in_input_and_output_for_functions,
    handle_transpose_in_pad,
    store_config_info,
)
from ..translations.data.object_like import BaseObjectLike
from .type_utils import Types


CONVERSION_OPTIONS = "not_s2s"


class TranslationOptions:
    """
    A container for defining translation flags for an object being translated source2source.

    Attributes:
        not_convert(bool): An attribute indicating that the object won't be translated.

    """

    def __init__(self, not_convert=False):
        self.not_convert = not_convert

    def attach(self, obj):
        if inspect.ismethod(obj):
            obj = obj.__func__

        if isinstance(obj, (types.FunctionType, type)):
            setattr(obj, CONVERSION_OPTIONS, self)
        else:
            raise exceptions.InvalidObjectException(
                f"Only function or class objects are supported but the passed in object has type {type(obj)}",
                propagate=True,
            )


def not_translate(obj=None):
    """
    A Decorator to suppress the translation of an object.

    Args:
        func(callable): The function to decorate.

    Returns:
        callable: A function which won't be translated in Source2Source.

    """
    options = TranslationOptions(not_convert=True)
    options.attach(obj)
    return obj


handle_get_item = not_translate(handle_get_item)
handle_set_item = not_translate(handle_set_item)

# ivy_repo/ivy/functional/frontends/torch/hub/hub.py
from ivy.functional.frontends.torch.hub import load_state_dict_from_url

load_state_dict_from_url = not_translate(load_state_dict_from_url)
