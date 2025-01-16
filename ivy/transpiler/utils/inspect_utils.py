# global
import gast
import functools
import inspect
import textwrap
from typing import Union
from types import FunctionType, MethodType

# local
from ..transformations import transformer_globals as glob
from ..exceptions.exceptions import (
    InvalidSourceException,
    InvalidObjectException,
)


def get_closure_vars(func):
    if not inspect.isfunction(func):
        return {}
    if func.__closure__ is None:
        return {}

    return dict(
        zip(func.__code__.co_freevars, (c.cell_contents for c in func.__closure__))
    )


def get_globals_from_module(obj):
    file = inspect.getfile(obj)

    with open(file, "r") as f:
        module = gast.parse(f.read())

    from ivy.transpiler.utils.ast_utils import ast_to_source_code

    globals = [node for node in module.body if isinstance(node, (gast.Assign))]
    globals_code = [ast_to_source_code(node) for node in globals]

    return globals_code


def recover_globals_attribute(src_obj, dst_obj):
    attr_name = "__globals__"

    src_globals = getattr(src_obj, attr_name, {})
    dst_globals = getattr(dst_obj, attr_name, {})

    for k, v in src_globals.items():
        # ignore builtin attribute.
        if not (k.startswith("__") and k.endswith("__")):
            dst_globals[k] = v


def object_to_source_code(function, dedent=True, handle_partial_mixed_lambdas=False):
    """
    Transforms func-like or type-like into raw string of source code.
    """
    if function is None:
        return ""

    if isinstance(function, functools.partial):
        function = function.func
    if not (
        inspect.isfunction(function)
        or inspect.ismethod(function)
        or inspect.isclass(function)
    ):
        raise TypeError(
            "The type of 'function' should be a function or method, but received {}.".format(
                type(function).__name__
            )
        )

    source_code_list, _ = inspect.getsourcelines(function)
    # Replace comments with blank lines so that error messages are not misplaced
    source_code_list = [
        line if not line.lstrip().startswith("#") else "\n" for line in source_code_list
    ]
    source_code = "".join(source_code_list)

    # Check if it's a lambda function and has `.partial_mixed_handler` assignment
    # target in the source code, then we need to make sure we only return the
    # lambda function definition itself rather than the rest of the assignment
    # targets.
    # function.__name__ will be '<lambda>' for lambdas
    if (
        handle_partial_mixed_lambdas
        and getattr(function, "__name__", "") == "<lambda>"
        and ".partial_mixed_handler" in source_code
    ):
        # Extract only the lambda expression part by splitting after the first '='
        split_code = source_code.split("=")
        if len(split_code) > 1:
            # Take the part after the first '=' and strip any surrounding spaces
            source_code = "=".join(split_code[1:]).strip()
        else:
            raise ValueError("Lambda source code does not have an assignment operator.")

    # Check for decorators in the source code
    existing_decorators = set(
        line.strip() for line in source_code.split("\n") if line.strip().startswith("@")
    )

    # Check for decorators and add them to the source code
    decorators = [
        "@" + dec
        for dec in glob.ALL_IVY_DECORATORS
        if hasattr(function, dec) and "@" + dec not in existing_decorators
    ]
    if function.__name__ == "asarray":
        decorators = [
            dec
            for dec in decorators
            if dec.replace("@", "@_asarray_") not in existing_decorators
        ]
    decorator_source = "\n".join(decorators) + "\n" if decorators else ""

    if dedent:
        source_code = textwrap.dedent(source_code)

    return decorator_source + source_code


def get_base_classes(cls):
    base_classes = []
    for base in cls.__bases__:
        if (
            base is not object
        ):  # Exclude the 'object' class which is a base for all classes
            base_classes.append(base)
            base_classes.extend(get_base_classes(base))
    return base_classes


def group_classes(classes):
    # Create a dictionary mapping each class to its base classes
    class_to_bases = {cls: get_base_classes(cls) for cls in classes}
    groups = []

    for cls in classes:
        if any(cls in group for group in groups):
            continue

        group = [cls]
        for base in class_to_bases[cls]:
            if base in classes:
                group.append(base)
        groups.append(group)

    # Filter out groups that are subsets of other groups
    groups = [
        group
        for group in groups
        if not any(set(group) < set(other_group) for other_group in groups)
    ]

    return groups


def _validate_object(object: Union[FunctionType, MethodType, type], source: str):
    """
    Validate and retrieve the appropriate object based on its source framework.
    """

    # Check if the object belongs to the Torch framework
    if hasattr(object, "__module__") and object.__module__.startswith("torch"):
        if source == "torch":
            return object
        raise InvalidSourceException(
            f"The object {object} is from the Torch framework, but the source is set to '{source}'.",
            propagate=True,
        )

    # Check if the object belongs to the Ivy framework
    if (
        hasattr(object, "__module__") and
        object.__module__.startswith("ivy.") and
        "ivy.transpiler" not in object.__module__
    ):
        if source == "ivy":
            return object
        raise InvalidSourceException(
            f"The object {object} is from the Ivy framework, but the source is set to '{source}'.",
            propagate=True,
        )

    # Check if the object is an instance and not a class itself
    if not (
        inspect.isclass(object)
        or inspect.ismethod(object)
        or isinstance(object, FunctionType)
    ):
        obj_type = type(object).__name__
        raise InvalidObjectException(
            f"`ivy.transpile` expected a class or a function, but received an instance of type '{obj_type}'. "
            "If you want to transpile an instantiated object, please pass its class instead. "
            "For restoring weights, consider using `ivy.sync_models` after instantiating "
            "the transpiled class to ensure correct weight transfer.",
            propagate=True,
        )

    # Check if the object's source code is accessible
    try:
        inspect.getsource(object)
    except OSError as e:
        raise InvalidObjectException(
            f"Unable to access the source code of the object '{object.__name__}' "
            f"of type '{type(object).__name__}'. This may occur if the object is dynamically "
            "created, defined in a Jupyter notebook, or not saved to disk.\n\n"
            "Suggested actions:\n"
            "- Ensure that the object is defined in a Python module or script that is saved to disk.\n"
            "- If you're working in a Jupyter notebook, consider using the `%%writefile` magic command to save the code to a file.\n"
            "  For example:\n"
            "  ```\n"
            "  %%writefile my_module.py\n"
            "  class MyClass:\n"
            "      def __init__(self):\n"
            "          pass\n"
            "  ```\n"
            "- After saving the object to a file, you can then import it and call `ivy.transpile`.\n"
            "- If the object is dynamically created, consider saving its definition to disk before transpiling.\n"
            f"Original error: {e}",
            propagate=True,
        )
    # Return the object as-is if all checks pass
    return object
