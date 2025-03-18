# global
import gast
import inspect
import ivy
import re

# local
from ....utils.api_utils import (
    get_function_from_modules,
    is_ivy_functional_api,
    is_backend_api,
    is_native_backend_api,
    SUPPORTED_BACKENDS_PREFIX,
)
from ....utils.ast_utils import (
    property_to_func,
    ast_to_source_code,
)
from ....utils.origin_utils import ORIGI_INFO
from ....utils.conversion_utils import is_builtin_function
from ..recursive_transformer.base_transformer import (
    BaseRecurser,
)
from ...transformer import Transformer
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ... import transformer_globals as glob


def func_from_call_node(node):
    """
    Checks whether the given node comes from a gast.Call node.
    Args:
        node (gast.AST): The node to check.
    Returns:
        bool: True if the node comes from a gast.Call node, False otherwise.
    """
    source_code = getattr(node, ORIGI_INFO).source_code.strip()
    try:
        parsed_code = gast.parse(source_code)
        # Check if the parsed code contains a gast.Call node
        return any(isinstance(n, gast.Call) for n in gast.walk(parsed_code))
    except Exception:
        return False


class IvyRecurser(BaseRecurser):
    """
    The `IvyRecurser` is another concrete class that inherits from `BaseRecurser`.
    It implements the transformation logic specific to `ivy` function calls and properties.
    It overrides the abstract methods from the base class to provide the necessary transformations.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        super(IvyRecurser, self).__init__(
            root=root, transformer=transformer, configuration=configuration
        )
        self._target_backend = ivy.current_backend()
        self._target_backend_func_str = (
            f"__{self.transformer.configuration.target}_backend_"
        )
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

    def _should_inject_backend_call(self, node):
        if (
            isinstance(node.func, gast.Attribute)
            and isinstance(node.func.value, gast.Call)
            and ast_to_source_code(node.func.value.func).strip()
            in ("current_backend", "ivy.utils.backend.current_backend")
            and getattr(self._target_backend, node.func.attr, None)
        ):
            return True
        return False

    def _get_function(self, name_str, modules, node):
        pattern = r"current_backend\([^)]*\)\.\w+"
        if name_str in glob.IVY_DECORATORS_TO_TRANSLATE or bool(
            re.search(pattern, name_str)
        ):
            modules = [self._target_backend, self._target_backend.__dict__["creation"]]
            name_str = name_str.split(".")[-1]
        orig_obj = get_function_from_modules(name_str, modules)
        # Retrieve the function associated with the property
        if isinstance(orig_obj, property):
            orig_obj = property_to_func(orig_obj, node)
        return orig_obj

    def _is_base_class(self, base):
        return "ivy.stateful.module" in base.__module__

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

    def _should_recurse_into_func(self, func_str, orig_func, to_ignore=()):
        # Part of the functional api
        if is_ivy_functional_api(orig_func):
            if any(m in func_str for m in to_ignore):
                return False
            return True
        # part of the backend api
        elif is_backend_api(orig_func):
            if any(func_str.__module__.startswith(m) for m in ("tf.", "tensorflow.")):
                return False
            return True

        elif is_native_backend_api(orig_func):
            return False
        # custom written function that might potentially contain other
        # functions we want to recurse into
        else:
            return True

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
                "source_to_source_translator",
                "profiling",
            )
        ):
            return False

        # Check if the function call is part of the functional API of Ivy or the standard libraries
        if (
            orig_func is None
            or not callable(orig_func)
            or inspect.isbuiltin(orig_func)
            or is_builtin_function(orig_func)
            or hasattr(orig_func, "__module__")
            and any(
                [
                    orig_func.__module__.startswith(m)
                    for m in (
                        "ivy.utils.backend",
                        "ivy.utils.exceptions",
                    )
                ]
            )
            or hasattr(orig_func, "__module__")
            and any(
                orig_func.__module__.startswith(fw) for fw in SUPPORTED_BACKENDS_PREFIX
            )  # no need to recurse into the native framework API at the ivy level
            or inspect.isclass(orig_func)
            and any(orig_func is cls for cls in self._ivy_classes)
            or inspect.isfunction(orig_func)
            and not self._should_recurse_into_func(
                func_str,
                orig_func,
                to_ignore=("current_backend_str",),
            )
        ):
            return False

        # Check if the object's source code is not available
        try:
            inspect.getsource(orig_func)
        except (TypeError, OSError):
            return False

        return True
