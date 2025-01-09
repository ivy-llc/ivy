# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module contains the user- and codegen-facing API for control_flow_experimental.autograph_ivy."""

import functools
import importlib
import inspect
import sys
import ivy
from types import FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType

from ..converters import break_statements
from ..converters import call_trees
from ..converters import conditional_expressions
from ..converters import continue_statements
from ..converters import control_flow
from ..converters import functions
from ..converters import return_statements
from ..converters import list_comprehensions
from ..converters import lists
from ..converters import slices
from . import list_ops
from . import unsupported_features_checker
from ..pyct import anno
from ..pyct import cfg
from ..pyct import inspect_utils
from ..pyct import qual_names
from ..pyct import transpiler
from ..pyct.static_analysis import activity
from ..pyct.static_analysis import (
    reaching_definitions,
)


# Actual source code transformation
#
#######################################################
class PyToIvy(transpiler.PyToPy):
    """The Ivy autograph_ivy transformer."""

    def __init__(self):
        super(PyToIvy, self).__init__()
        self._extra_locals = None

    def get_transformed_name(self, node):
        return "ivy__" + super(PyToIvy, self).get_transformed_name(node)

    def get_extra_locals(self):
        if self._extra_locals is None:
            # TODO(mdan): Move into core or replace with an actual importable module.
            # Craft a module that exposes the external API as well as certain
            # internal modules.
            module_spec = importlib.machinery.ModuleSpec("autograph_ivy", None)
            ag_internal = importlib.util.module_from_spec(module_spec)
            ag_internal.__dict__.update(inspect.getmodule(PyToIvy).__dict__)
            ag_internal.__dict__.update(list_ops.__dict__)
            # TODO(mdan): Add safeguards against name clashes.
            self._extra_locals = {"ivy__": ag_internal, "ivy": ivy}
        return self._extra_locals

    def initial_analysis(self, node, ctx):
        graphs = cfg.build(node)
        node = qual_names.resolve(node)
        node = activity.resolve(node, ctx, None)
        node = reaching_definitions.resolve(node, ctx, graphs)
        anno.dup(
            node,
            {
                anno.Static.DEFINITIONS: anno.Static.ORIG_DEFINITIONS,
            },
        )
        return node

    def transform_ast(self, node, ctx):
        unsupported_features_checker.verify(node)
        node = self.initial_analysis(node, ctx)

        node = functions.transform(node, ctx)
        node = break_statements.transform(node, ctx)
        # Note: sequencing continue canonicalization before for loop one avoids
        # dealing with the extra loop increment operation that the for
        # canonicalization creates.
        node = continue_statements.transform(node, ctx)
        node = return_statements.transform(node, ctx)
        # node = lists.transform(node, ctx)
        # node = slices.transform(node, ctx)
        node = call_trees.transform(node, ctx)
        node = control_flow.transform(node, ctx)
        node = conditional_expressions.transform(node, ctx)
        node = list_comprehensions.transform(node, ctx)
        return node


def return_none_or_val(retval, do_return):
    if do_return:
        return retval
    else:
        return None


def _is_of_known_loaded_module(f, module_name):
    mod = sys.modules.get(module_name, None)
    if mod is None:
        return False
    if any(v is not None for v in mod.__dict__.values() if f is v):
        return True
    return False


def _is_known_loaded_type(f, module_name, entity_name):
    """Tests whether the function or method is an instance of a known type."""
    if module_name not in sys.modules or not hasattr(
        sys.modules[module_name], entity_name
    ):
        return False
    type_entity = getattr(sys.modules[module_name], entity_name)
    if isinstance(f, type_entity):
        # The method if of this type. Example:
        #
        # o = ClassType()
        # function(o.method)()
        return True

    if inspect.ismethod(f):
        # The unbound method if of this type. Example:
        #
        # class ClassType:
        #     @function
        #     def method(self):
        #         ...
        # o = ClassType()
        # o.method()
        if isinstance(f.__func__, type_entity):
            return True
    return False


def is_unsupported(o):
    """Checks whether an entity is supported."""

    if _is_known_loaded_type(o, "functools", "_lru_cache_wrapper"):
        return True

    if inspect_utils.isconstructor(o):
        return True

    if any(
        _is_of_known_loaded_module(o, m)
        for m in ("ivy", "torch", "tf", "jax", "haiku", "paddle")
    ):
        return True

    return False


def is_user_defined(func):
    # Check if the function is a built-in function
    if inspect.isbuiltin(func):
        return False

    # Check if the function is one of the specified functions
    if func.__module__ in ["ivy", "tf", "torch", "jax", "haiku", "paddle"]:
        return False

    # If none of the above conditions are met, the function is user-defined
    return True


def converted_call(f, args, kwargs):
    """Converts a function call inline.

    For internal use only.

    Note: The argument list is optimized for readability of generated code, which
    may look like this:

        converted_call(f, (arg1, arg2), None, fscope)
        converted_call(f, (), dict(arg1=val1, **kwargs), fscope)
        converted_call(f, (arg1, arg2) + varargs, dict(**kwargs), lscope)

    Args:
        f: The function to convert.
        args: Tuple, the original positional arguments of f
        kwargs: Optional[Dict], the original keyword arguments of f
        caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
            scope of the converted function in which this call was originally made.
        options: Optional[converter.ConversionOptions], conversion options. If not
            specified, the value of caller_fn_scope.callopts is used. Either options
            or caller_fn_scope must be present.

    Returns:
        Any, the result of executing a possibly-converted `f` with the given
            arguments.
    """
    # If this is a partial, unwrap it and redo all the checks.
    if isinstance(f, functools.partial):
        new_kwargs = {}
        if f.keywords is not None:
            # Use copy to avoid mutating the underlying keywords.
            new_kwargs = f.keywords.copy()
        if kwargs is not None:
            new_kwargs.update(kwargs)
        new_args = f.args + args
        return converted_call(
            f.func,
            new_args,
            new_kwargs,
        )

    # If the function is wrapped, we don't need to go inside of it.
    if is_user_defined(f) or hasattr(f, "wrapped_for_tracing"):
        if kwargs:
            return f(*args, **kwargs)
        else:
            return f(*args)

    if is_unsupported(f):
        return _call_unconverted(f, args, kwargs)

    # if inspect_utils.isbuiltin(f) or str(f).__contains__('ivy') or hasattr(f,"__wrapped__"):
    #     if kwargs:
    #         return f(*args, **kwargs)
    #     else:
    #         return f(*args)

    if inspect.ismethod(f) or inspect.isfunction(f):
        target_entity = f
        effective_args = args

    elif hasattr(f, "__class__") and hasattr(f.__class__, "__call__"):
        target_entity = f.__class__.__call__
        effective_args = (f,) + args

    else:
        raise NotImplementedError('unknown callable type "%s"' % type(f))

    if not hasattr(target_entity, "__code__"):
        return _call_unconverted(f, args, kwargs)
    elif (
        hasattr(target_entity.__code__, "co_filename")
        and target_entity.__code__.co_filename == "<string>"
    ):
        return _call_unconverted(f, args, kwargs)

    converted_f = to_functional_form(target_entity)
    if kwargs is not None:
        result = converted_f(*effective_args, **kwargs)
    else:
        result = converted_f(*effective_args)

    return result


def _call_unconverted(f, args, kwargs):
    """Calls the original function without converting."""

    if inspect.ismethod(f):
        return f.__self__.call(args, kwargs)

    if kwargs is not None:
        return f(*args, **kwargs)
    return f(*args)


def to_functional_form(entity, program_ctx=None):
    """Applies autograph_ivy to entity."""

    if isinstance(entity, (BuiltinFunctionType, BuiltinMethodType)):
        return entity

    functionlike = (
        FunctionType,
        MethodType,
    )
    if (
        hasattr(entity, "__call__")
        and callable(entity)
        and not isinstance(entity, functionlike)
    ):
        if entity.__call__ is not entity:
            new_call = to_functional_form(entity.__call__)
            entity.__call__ = new_call
            return entity

    # TODO(mdan): Put these extra fields inside __autograph_ivy_info__.
    if not hasattr(entity, "__code__"):
        raise ValueError(
            "Cannot apply autograph_ivy to a function that doesn't "
            "expose a __code__ object. If this is a @tf.function,"
            " try passing f.python_function instead."
        )

    transformed, module, source_map = PyToIvy().transform(entity, program_ctx)

    transformed.ivy_module = module
    transformed.ivy_source_map = source_map

    if hasattr(entity, "__self__"):
        original = transformed

        @functools.wraps(original)
        def wrapped_transformed(*args, **kwargs):
            return original(entity.__self__, *args, **kwargs)

        transformed = wrapped_transformed

    return transformed
