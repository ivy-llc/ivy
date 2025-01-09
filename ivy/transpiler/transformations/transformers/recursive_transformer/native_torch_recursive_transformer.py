# global
import gast
import inspect

# local
from ....utils.conversion_utils import is_builtin_function
from ....utils.api_utils import get_function_from_modules
from ..recursive_transformer.base_transformer import (
    BaseRecurser,
)
from ...transformer import Transformer
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)


class NativeTorchRecurser(BaseRecurser):
    """
    The `NativeTorchRecurser` is a concrete class that inherits from `BaseRecurser`.
    It implements the transformation logic specific to native PyTorch classes/functions.
    It overrides the abstract methods from the base class to provide the necessary transformations.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        super(NativeTorchRecurser, self).__init__(
            root=root, transformer=transformer, configuration=configuration
        )
        try:
            import torch  # noqa
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "`torch` was not found installed on your system. Please proceed "
                "to install it and restart your interpreter to see the changes."
            ) from exc

        self._native_classes = [
            torch.nn.Module,
            torch.nn.Parameter,
            torch.Tensor,
            torch.Size,
        ]

    def _get_function(self, name_str, module, node):
        return get_function_from_modules(name_str, module)

    def _is_base_class(self, base):
        return base.__module__ == "torch.nn.modules.module"

    def _get_module_node(self):
        return gast.Attribute(
            value=gast.Name(
                id="nn", ctx=gast.Load(), annotation=None, type_comment=None
            ),
            attr="Module",
            ctx=gast.Load(),
        )

    def _get_module_name(self):
        return "nn.Module"

    def _should_translate(self, func_str, orig_func):

        # Check if the function call is one of ivy.utils.decorator_utils, and if so, translate
        if (
            hasattr(orig_func, "__module__")
            and orig_func.__module__ == "ivy.utils.decorator_utils"
        ):
            return True

        if any(
            name in func_str
            for name in (
                "profiling_timing_decorator",
                "profiling_logging_decorator",
                "decorated",
                "profiling",
            )
        ):
            return False

        # Check if the function call is part of the functional API of torch or the standard libraries
        if (
            orig_func is None
            or not callable(orig_func)
            or not hasattr(orig_func, "__module__")
            or inspect.isbuiltin(orig_func)
            or is_builtin_function(orig_func)
            or (
                inspect.isclass(orig_func)
                and any(orig_func is cls for cls in self._native_classes)
            )
            or any(
                [
                    name in func_str
                    for name in (
                        "nn.functional.",
                        "F.",
                        "_C.",
                        "jit.",
                        "fx.",
                        "self",
                    )
                ]
            )
            or any(
                [
                    m in orig_func.__module__
                    for m in (
                        "torch.utils",
                        "torch.autograd",
                        "torch._C",
                        "torch.functional",
                        "torch.nn.functional",
                        "torch.random",
                        "torch.amp",
                        "torch.cuda",
                        "torch.xpu",
                        "torch.hub",
                        "torch.backends",
                        "torch.overrides",
                        "torch.distributed",
                        "torch._utils_internal",
                        "torch._dynamo",
                        "torch.serialization",
                        "torch.nn.utils.rnn",
                        "torch.onnx",
                        # other native framework functions are sometimes present in the sourcecode too;
                        # we also don't want to recurse into these
                        "jax._src",
                        "tensorflow.python",
                    )
                ]
                + [orig_func.__module__ in ["jax", "numpy", "tensorflow", "torch"]]
            )
        ):
            return False

        # Check if the object's source code is not available
        try:
            inspect.getsource(orig_func)
        except (TypeError, OSError):
            return False

        return True
