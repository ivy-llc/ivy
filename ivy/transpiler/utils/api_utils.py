# global
from collections.abc import Iterable
import importlib
import inspect
import ivy
import gast
import os
import types
from .type_utils import Types

FRONTEND_ARRAY_MODULES = [
    "ivy.functional.frontends.torch.tensor",
]
IVY_MODULE_PREFIX = "ivy."
SUPPORTED_BACKENDS_PREFIX = [
    "torch",
    "tf",
    "tensorflow",
    "keras",
    "jax",
    "jnp",
    "jaxlib",
    "numpy",
    "np",
    "paddle",
]
SUPPORTED_BACKENDS_MODULE_STRS = {
    "tensorflow": {True: "tensorflow.keras.Model", False: "tensorflow.keras.Layer"},
    "jax": {True: "flax.nnx.Module", False: "flax.nnx.Module"},
    "numpy": {True: "type", False: "type"},
}
TRANSLATED_OBJ_PREFIX = [
    "Translated_",
    "ivy_",
    "tensorflow_",
    "torch_",
    "jax_",
    "numpy_",
    "paddle_",
    "ivy.compiler._cache",
]
# set of suffixes that are appended to the translated fn name
TRANSLATED_OBJ_SUFFIX = [
    "frnt",
    "frnt_",
    "bknd",
    "bknd_",
]
SOURCE_TO_SOURCE_PREFIX = "source_to_source_translator"
# TODO: need to find a way to resolve name conflicts. i.e: x.add(..) could either
# refer to `torch.Tensor.add` or `set.add`
# Currently, we use a priority-based scheme to resolve name clashes, where builtins
# have higher priority than frontends. This implies that x.add(..) is interpreted as
# set.add and **NOT** torch.Tensor.add
BUILTIN_CLASSES = [list, tuple, dict, set, frozenset, str, complex]


def get_native_array_str_from_backend(backend_str: str):
    if isinstance(ivy.NativeArray, tuple):
        return ivy.NativeArray[0].__name__
    else:
        return ivy.NativeArray.__name__


def get_native_module_str_from_backend(backend_str: str, is_root_obj: bool, depth: int):
    is_root_obj = is_root_obj or depth == 0
    return SUPPORTED_BACKENDS_MODULE_STRS[backend_str][is_root_obj]


def get_function_from_modules(function_name: str, modules: Iterable):
    assert isinstance(
        modules, Iterable
    ), f"modules must be an iterable but is of type {type(modules)}"
    function_parts = function_name.split(".")

    # Case 1: <mod_chain>.fn --> try to get the <mod_chain> from "modules" and then get fn from <mod_chain>
    # +
    # Case 2: fn --> try to get fn from "modules"
    for module in modules:
        skip_outer_loop = False
        try:
            # case 1: traverse the <mod_chain>
            func = module.__dict__[function_parts[0]]
            for part in function_parts[1:]:
                try:
                    func = getattr(func, part)
                except AttributeError:
                    try:
                        # case 2: directly try to get the function from the module
                        func = getattr(module, function_parts[-1])
                        break
                    except AttributeError:
                        skip_outer_loop = True
                        break

            if skip_outer_loop:
                continue  # Skip outer loop

            if func is None:
                # in the case the global is None, we need to return a string -
                # returning None would be saying the target obj is invalid
                return "None"
            return func
        except KeyError:
            continue

    # Case 3: <mod_chain>.fn --> try to get <mod_chain> from importlib and then get fn from <mod_chain>
    if len(function_parts) > 1:
        *mod_list, obj = function_parts
        mod = ".".join(mod_list)
        try:
            module = importlib.import_module(mod)
            try:
                return getattr(module, obj)
            except AttributeError:
                pass
        except ModuleNotFoundError:
            pass

    # if all cases fail, return None
    return None


def get_frontend_class(class_name):
    mod, cls = class_name.split(".")
    module = importlib.import_module(f"ivy.functional.frontends.{mod}")
    return getattr(module, cls)


def get_hf_class(class_name):
    try:
        module = importlib.import_module(f"transformers.modeling_utils")
    except (TypeError, AttributeError, ModuleNotFoundError):
        return None
    return getattr(module, class_name)


def maybe_get_methods(cls):
    methods = []
    for name, obj in inspect.getmembers(cls):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            methods.append(name)
    return methods


def maybe_get_properties(cls):
    methods = []
    for name, obj in inspect.getmembers(cls):
        if is_property(name, cls):
            methods.append(name)
    return methods


def maybe_get_frontend_base_module_source(bases):
    for index, base in enumerate(bases):
        if is_frontend_stateful_api(base):
            return (index, inspect.getsource(inspect.getmodule(base)))
    return -1, None


def maybe_get_frontend_base_methods(obj):
    base_methods = inspect.getmembers(
        obj,
        lambda x: (
            inspect.isfunction(x)
            and is_frontend_stateful_api(x)
            and (x.__name__.startswith("__") or x.__name__ == "to")
        ),
    )
    method_names = [method[0] for method in base_methods]
    method_sources = [inspect.getsource(method[1]) for method in base_methods]
    return {name: source for (name, source) in zip(method_names, method_sources)}


def is_builtin_method(method, to_ignore=()):
    for cls in BUILTIN_CLASSES:
        if method not in to_ignore and method in dir(cls):
            return True
    return False


def is_method_of_class(method_name, cls, to_ignore=()):
    from ivy.transpiler.translations.data.object_like import BaseObjectLike

    if isinstance(cls, BaseObjectLike):
        return method_name in cls.methods

    if method_name in dir(cls):
        method_ = getattr(cls, method_name)
        return (
            isinstance(method_, types.FunctionType)
            and (hasattr(method_, "__module__") and method_.__module__ not in to_ignore)
        ) or is_property(method_name, cls)
    return False


def is_property(method_name, cls, to_ignore=()):
    from ivy.transpiler.translations.data.object_like import BaseObjectLike

    if isinstance(cls, BaseObjectLike):
        return method_name in cls.properties
    try:
        method_ = getattr(cls, method_name)
        return isinstance(method_, property) and all(
            [s not in method_name for s in to_ignore]
        )
    except (TypeError, AttributeError):
        return False


def is_internal_api(obj):
    # Note: A api in module source_to_source is not a real Source2Source api.
    return is_api_in_module(obj, SOURCE_TO_SOURCE_PREFIX)


def is_api_in_module(obj, module_prefix):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)


def is_ivy_api(obj):
    return is_api_in_module(obj, IVY_MODULE_PREFIX) and not is_api_in_module(
        obj, "ivy.compiler._cache"
    )  # filter for cached translated objs


def is_native_backend_api(obj):
    return any(is_api_in_module(obj, prefix) for prefix in SUPPORTED_BACKENDS_PREFIX)


def is_translated_api(obj):
    return any(
        is_api_in_module(obj, prefix) for prefix in TRANSLATED_OBJ_PREFIX
    ) or is_api_in_module(
        obj, "ivy.compiler._cache"
    )  # filter for cached translated objs


def is_backend_api(obj):
    return any(
        is_api_in_module(obj, f"ivy.functional.backends.{prefix}")
        for prefix in SUPPORTED_BACKENDS_PREFIX
    )


def is_frontend_api(obj):
    return any(
        is_api_in_module(obj, f"ivy.functional.frontends.{prefix}")
        for prefix in SUPPORTED_BACKENDS_PREFIX
    )


def is_ivy_functional_api(obj):
    if callable(obj):
        return is_ivy_api(obj) or is_native_backend_api(obj)
    return False


def is_frontend_stateful_api(obj):
    # TODO: Make this generic to work with all frontend stateful apis
    # not only just torch
    return is_api_in_module(obj, "ivy.functional.frontends.torch.nn.module")


def is_subclass_of_frontend_stateful_api(obj):
    # TODO: Make this generic to work with all frontend stateful apis
    # not only just torch
    return any(
        is_api_in_module(b, "ivy.functional.frontends.torch.nn.module")
        for b in obj.__bases__
    )


def is_hf_pretrained_class(obj):
    return hasattr(obj, "__bases__") and any(
        "pretrainedmodel" in base.__name__.lower() for base in obj.__bases__
    )


def is_mixed_function(fn):
    return hasattr(fn, "compos")


def is_helper_func(object_like):
    return object_like.type == Types.FunctionType and not object_like.is_root_obj


def is_submodule_of(object_like, cls_name):
    mro_list = object_like._derive_mro()
    return any(cls_name in str(mro) for mro in mro_list)


def from_conv_block(func):
    """
    function to detect the presence of a convolution block
    (eg: torch.nn.Conv2d). It is utilized later in determining
    whether to apply certain transpose optimizations or not.
    """
    from ivy.utils.decorator_utils import CONV_BLOCK_FNS

    if hasattr(func, "__name__"):
        return any(fn in func.__name__ for fn in CONV_BLOCK_FNS)


def has_conv_args(args_node):
    """
    Checks if the arguments in the gast.arguments node contain 'data_format' or 'filter_format'.

    Args:
        args_node (gast.arguments): The gast node containing the arguments of a function.

    Returns:
        bool: True if 'data_format' or 'filter_format' is found among the argument names, False otherwise.
    """
    for arg in args_node.args:
        if isinstance(arg, gast.Name) and arg.id in {"data_format", "filter_format"}:
            return True

    for kwarg in args_node.kwonlyargs:
        if isinstance(kwarg, gast.Name) and kwarg.id in {
            "data_format",
            "filter_format",
        }:
            return True

    return False


def is_compiled_module(module_name: str) -> bool:
    """
    Whether a given module actually contains python source code, or whether it just contains compiled .so or .pyd files.

    Checks there are no .py files present in the module (other than __init__.py) and there are at least one .so or .pyd file.

    Args:
        module_name (str): The name of the module to check

    Returns:
        (bool): Whether the module only contains compiled code
    """

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return False

    # Get the module's directory
    if hasattr(module, "__path__"):
        module_dir = module.__path__[0]
    elif hasattr(module, "__file__"):
        module_dir = os.path.dirname(module.__file__)
    else:
        return False

    # Check for __init__.py file
    init_file_path = os.path.join(module_dir, "__init__.py")
    if not os.path.isfile(init_file_path):
        return False

    # Check for other .py files
    python_files = [
        f for f in os.listdir(module_dir) if f.endswith(".py") and f != "__init__.py"
    ]
    if python_files:
        return False

    # Check for .so or .pyd files
    compiled_files = [
        f for f in os.listdir(module_dir) if f.endswith(".so") or f.endswith(".pyd")
    ]
    if compiled_files:
        return True
    else:
        return False


def copy_module(module):
    ret = types.ModuleType(module.__name__, module.__doc__)
    ret.__class__ = type(module)
    ret.__dict__.update(module.__dict__)
    return ret
