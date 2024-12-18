# global
import gast
import importlib
import inspect
import ivy
import types

# local
from ....utils.api_utils import (
    get_function_from_modules,
    is_ivy_api,
    is_frontend_api,
    SUPPORTED_BACKENDS_PREFIX,
)
from ....utils.ast_utils import property_to_func
from ....utils.conversion_utils import is_builtin_function
from ..recursive_transformer.base_transformer import (
    BaseRecurser,
)

from ...transformer import Transformer
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)


class FrontendTorchRecurser(BaseRecurser):
    """
    The `FrontendTorchRecurser` is another concrete class that inherits from `BaseRecurser`.
    It implements the transformation logic specific to `torch frontend` function calls and properties.
    It overrides the abstract methods from the base class to provide the necessary transformations.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        super(FrontendTorchRecurser, self).__init__(
            root=root, transformer=transformer, configuration=configuration
        )
        import ivy.functional.frontends.torch as torch_frontend
        from ivy.functional.frontends.numpy.ndarray.ndarray import ndarray

        self._frontend = torch_frontend
        self._frontend_classes = [
            torch_frontend.Tensor,
            torch_frontend.nn.Parameter,
            torch_frontend.nn.Module,
            ndarray,
        ]
        self._ivy_classes = [
            ivy.Module,
            ivy.Container,
            ivy.Array,
            ivy.Dtype,
            ivy.IntDtype,
            ivy.UintDtype,
            ivy.FloatDtype,
            ivy.ComplexDtype,
            ivy.Device,
            ivy.Shape,
            ivy.NativeArray,
            ivy.NativeDevice,
            ivy.NativeDtype,
            ivy.NativeShape,
            ivy.ArrayMode,
            ivy.PreciseMode,
        ]

    def _get_function(self, name_str, modules, node):
        # try retrieving the function:
        # 1. via the frontends
        # 2. via the current module
        modules_to_search = modules
        if (
            "torch.Tensor" in name_str
            and self.transformer.object_like.module
            == self._frontend_classes[0].__module__
        ):
            name_str = name_str.replace(
                "torch.", ""
            )  # retrieve the frontend Tensor method via the Tensor namespace.
        orig_obj = get_function_from_modules(name_str, modules_to_search)

        # Retrieve the function associated with the property
        if isinstance(orig_obj, property):
            orig_obj = property_to_func(orig_obj, node)

        return orig_obj

    def _is_base_class(self, base):
        return any(
            substr in base.__module__
            for substr in ("torch.nn.modules.module", "numpy.ndarray.ndarray")
        )

    def _is_method_of_frontend_tensor_class(self, name_str, orig_func):
        return orig_func is not None and any(
            getattr(_cls, name_str.split(".")[-1], lambda x: x) is orig_func
            for _cls in self._frontend_classes[:-1]
        )

    def _get_module_node(self):
        return gast.Attribute(
            value=gast.Name(
                id="ivy", ctx=gast.Load(), annotation=None, type_comment=None
            ),
            attr="Module",
            ctx=gast.Load(),
        )

    def _get_module_name(self):
        return "ivy.Module"

    def _should_translate(self, func_str, orig_func):
        # Check if the function call is from frontend torch.Tensor or torch.nn.Parameter
        # and if so, translate
        if self._is_method_of_frontend_tensor_class(func_str, orig_func):
            return True

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
                "source_to_source_translator",
                "profiling",
            )
        ):
            return False

        # Check if the function call is part of the functional API of torch or the standard libraries
        if (
            orig_func is None
            or not callable(orig_func)
            or inspect.isbuiltin(orig_func)
            or is_builtin_function(orig_func)
            or any(
                [
                    name in func_str
                    for name in (
                        "cuda.",
                        "fx.",
                        "self",
                        # "ivy.",
                    )
                ]
            )
            or hasattr(orig_func, "__module__")
            and any(
                orig_func.__module__.startswith(fw) for fw in SUPPORTED_BACKENDS_PREFIX
            )  # no need to recurse into the native frameworks API at the frontend level
            or (
                inspect.isclass(orig_func)
                and any(
                    orig_func is cls
                    for cls in (self._frontend_classes + self._ivy_classes)
                )
            )
            or hasattr(orig_func, "__module__")
            and is_ivy_api(orig_func)
            and not is_frontend_api(orig_func)
            or hasattr(orig_func, "__qualname__")
            and orig_func.__qualname__ == "Tensor"
        ):
            if any(
                func_str.startswith(mod) and func_str != mod
                for mod in SUPPORTED_BACKENDS_PREFIX
            ):
                orig_func = get_function_from_modules(
                    func_str, [importlib.import_module("torch")]
                )
                frontend_func = get_function_from_modules(func_str, [self._frontend])
                if (
                    orig_func is not None
                    and frontend_func is None
                    and callable(orig_func)
                ):
                    self.transformer.missing_frontends.append(func_str)
            return False

        # Check if the object's source code is not available
        try:
            inspect.getsource(orig_func)
        except (TypeError, OSError):
            return False

        return True
