# global
import os
import functools
import importlib
import atexit
import inspect
import sys
import threading
import builtins

import ivy
import gast
import builtins
from typing import Any, Callable
from types import FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType

# local
from ..control_flow_experimental.autograph_ivy.pyct import parser
from ..control_flow_experimental.autograph_ivy.pyct import inspect_utils
from ..control_flow_experimental.autograph_ivy.pyct import qual_names
from ..control_flow_experimental.autograph_ivy.pyct import templates
from ..control_flow_experimental.autograph_ivy.converters import functions
from ..control_flow_experimental.autograph_ivy.operators import py_builtins
from ..control_flow_experimental.autograph_ivy import operators
from ..control_flow_experimental.autograph_ivy.pyct import (
    anno,
    cache,
    cfg,
    loader,
    transpiler as py_transpiler,
)
from ..control_flow_experimental.autograph_ivy.core import (
    converter,
    function_wrappers,
    unsupported_features_checker,
)
from ..control_flow_experimental.autograph_ivy.pyct.static_analysis import (
    activity,
    reaching_definitions,
)
from ..control_flow_experimental.autograph_ivy.converters.call_trees import (
    _ArgTemplateBuilder,
)
from .. import globals as glob


# Globals #
# ------- #

_MAX_CACHE_SIZE = 300
_TRANSFORMED_FN_CACHE = cache.CodeObjectCache()
_FUNCTIONLIKE = (
    FunctionType,
    MethodType,
)
_BUILTIN_FUNCTIONLIKE = (
    BuiltinFunctionType,
    BuiltinMethodType,
)
_SET_TRACE_WARNED = False
_APIS_TO_IGNORE = (
    "ivy.func_wrapper",
    "ivy.functional.ivy",
    "ivy.data_classes",
    "ivy.stateful",
)
_BCKNDS_TO_IGNORE = (
    "ivy.functional.backends.torch",
    "ivy.functional.backends.tensorflow",
    "ivy.functional.backends.numpy",
    "ivy.functional.backends.jax",
    "ivy.functional.backends.paddle",
)
_FRNTNDS_TO_IGNORE = (
    "ivy.functional.frontends.torch",
    "ivy.functional.frontends.tensorflow",
    "ivy.functional.frontends.numpy",
    "ivy.functional.frontends.jax",
    "ivy.functional.frontends.paddle",
)
_FWS_TO_IGNORE = (
    "torch",
    "tensorflow",
    "numpy",
    "jax",
    "paddle",
    "collections",
    "pdb",
    "inspect",
    "copy",
)


# Helpers #
# ------- #


def _setattr_transformed(obj):
    """Sets the attribute on a transformed object"""
    try:
        setattr(obj, "ivy_builtins_transformed", True)
    except AttributeError:
        try:
            obj.ivy_builtins_transformed = True
        except AttributeError:
            pass


def _get_effective_args(f, args):
    """Returns the effective args for a function/method by handling
    the self object or for callable instances by returning their `__call__`
    method."""
    if inspect.ismethod(f) or inspect.isfunction(f):
        target_entity = f
        effective_args = args
        f_self = getattr(f, "__self__", None)

        if f_self is not None:
            effective_args = (f_self,) + effective_args

    elif hasattr(f, "__class__") and hasattr(f.__class__, "__call__"):
        # Callable objects. Dunder methods have special lookup rules, see:
        # https://docs.python.org/3/reference/datamodel.html#specialnames
        # TODO(mdan): Recurse into converted_call to simplify other verifications.
        # This should be handled in the same way as partials.
        target_entity = f.__class__.__call__
        effective_args = (f,) + args

    else:
        target_entity = f
        raise NotImplementedError('Unknown callable type "%s"' % type(f))
    return target_entity, effective_args


def _get_fn_from_path(path: str) -> Callable:
    """Returns the callable fn from a given path"""
    split_path = path.split(".")
    fn = importlib.import_module(split_path[0])
    for p in split_path[1:]:
        fn = getattr(fn, p)
    return fn


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


def _exposes_code(o):
    """Checks whether an entity exposes its __code__
    object or not."""
    return hasattr(o, "__code__") and o.__code__


def _is_ivy_module_call(o):
    """Checks whether an entity is the ivy.Module._call."""
    if _is_callable_funclike(o):
        # Check whether it contains a __self__ attr
        if (
            hasattr(o, "__self__")
            and isinstance(o.__self__, ivy.Module)
            and type(o.__self__) is not ivy.Module
            and o.__name__ == "_call"
        ):
            return True
    return False


def _is_callable_funclike(o):
    """Checks whether an entity is callable."""
    return hasattr(o, "__call__") and callable(o)


def _is_transformed(o):
    """Checks whether an entity is already transformed
    using our Autograph engine."""
    return hasattr(o, "ivy_builtins_transformed") and o.ivy_builtins_transformed


def _is_unsupported(o):
    """Checks whether an entity is supported."""
    module_to_type = {
        "functools": "_lru_cache_wrapper",
        "ivy": "Array",
        "ivy": "Container",
        "ivy": "Module",
        "ivy": "ModuleHelpers",
    }

    for module, obj_type in module_to_type.items():
        if _is_known_loaded_type(o, module, obj_type):
            return True

    if inspect_utils.isconstructor(o):
        return True

    if hasattr(o, "__module__") and o.__module__:
        derived_obj = any(
            fw in o.__module__
            for fw in _APIS_TO_IGNORE
            + _FRNTNDS_TO_IGNORE
            + _BCKNDS_TO_IGNORE
            + ("haiku", "flax")
        )
        root_obj = any(fw == o.__module__ for fw in _FWS_TO_IGNORE)
        return derived_obj or root_obj

    return False


def _isbuiltin(f):
    """Returns True if the argument is a built-in function."""
    if id(f) in inspect_utils._BUILTIN_FUNCTION_IDS:
        return True
    elif isinstance(f, BuiltinFunctionType):
        return True
    elif inspect.isbuiltin(f):
        return True
    elif f is eval:
        return True
    else:
        return False


def _call_builtin(f, args, kwargs):
    """Wraps the builtin function or method or calls it unconverted."""
    kwargs = {} if kwargs is None else kwargs
    if isinstance(f, BuiltinMethodType) and f.__name__ in glob.BUILTIN_METHODS_TO_TRACK:
        from tracer.helpers import wrapped_builtin_callable

        kwargs.update({"is_builtin_method": True})
        return wrapped_builtin_callable(f, *args, **kwargs)
    return _call_unconverted(f, args, kwargs)


def _call_unconverted(f, args, kwargs):
    """Calls the original function without converting."""
    if kwargs is not None:
        return f(*args, **kwargs)
    return f(*args)


def converted_call(f, args, kwargs, caller_fn_scope=None, options=None):
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
    # Sometimes a builtin method type gets called here which is contained within
    # the __wrapped__ attribute of a wrapping fuction. This wrapping function is
    # the one that is wrapped by our own WrappedCallable class but we somehow
    # lose reference to that so need to restore that reference here
    from tracer.wrapping import FUNC_TO_PATH

    if (
        isinstance(f, BuiltinFunctionType)
        and hash(f)
        and f in FUNC_TO_PATH
        and not hasattr(f, "wrapped_for_tracing")
    ):
        try:
            fn = _get_fn_from_path(FUNC_TO_PATH[f])
            if hasattr(fn, "wrapped_for_tracing"):
                f = fn
        except Exception as _:
            pass

    # If the function is wrapped, we don't need to go inside of it.
    if hasattr(f, "wrapped_for_tracing"):
        if kwargs:
            return f(*args, **kwargs)
        else:
            return f(*args)

    # Extract scope options
    if options is None:
        if caller_fn_scope is None:
            raise ValueError("either caller_fn_scope or options must have a value")
        options = caller_fn_scope.callopts

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
            caller_fn_scope=caller_fn_scope,
            options=options,
        )

    # Handle builtins and their scope contexts
    if _isbuiltin(f):
        if f is eval:
            return py_builtins.eval_in_original_context(f, args, caller_fn_scope)
        elif f is super:
            return py_builtins.super_in_original_context(f, args, caller_fn_scope)
        elif f is builtins.globals:
            return py_builtins.globals_in_original_context(caller_fn_scope)
        elif f is locals:
            return py_builtins.locals_in_original_context(caller_fn_scope)
        else:
            return _call_builtin(f, args, kwargs)

    # If fn is not supported, call uncoverted
    if _is_unsupported(f):
        return _call_unconverted(f, args, kwargs)

    # Get target fn and effective args
    try:
        target_entity, effective_args = _get_effective_args(f, args)
    except Exception as _:
        return _call_unconverted(f, args, kwargs)

    # Objects that don't expose their code object or co_filename can't be transformed
    if not hasattr(target_entity, "__code__"):
        return _call_unconverted(f, args, kwargs)
    elif (
        hasattr(target_entity.__code__, "co_filename")
        and target_entity.__code__.co_filename == "<string>"
    ):
        return _call_unconverted(f, args, kwargs)

    # Perform transformation if none of the checks above suffice
    try:
        program_ctx = converter.ProgramContext(options=options)
        converted_f = transform_builtins(target_entity, program_ctx)
    except Exception as _:
        # Fallback to unconverted
        return _call_unconverted(f, args, kwargs)

    # If the transformation returned the original object, call it
    if converted_f is target_entity:
        return _call_unconverted(f, args, kwargs)

    # Call the transformed object
    if kwargs is not None:
        try:
            result = converted_f(*effective_args, **kwargs)
        except Exception as _:
            # In case of failure, fallback to calling unconverted fn
            result = _call_unconverted(f, args, kwargs)
    else:
        try:
            result = converted_f(*effective_args)
        except Exception as _:
            result = _call_unconverted(f, args, kwargs)

    return result


# BuiltinsTransformer #
# ----------- #


class _Function(object):
    no_root = True

    def __init__(self):
        self.context_name = None


class BuiltinCallsTransformer(converter.Base):
    def __init__(self, fn_name, ctx, transpiling):
        self._fn_name = fn_name
        self._first_node = None
        self._import_inserted = False
        self._transpiling = transpiling
        self.contains_builtins = False
        self.contains_lazy_transforms = False
        super(BuiltinCallsTransformer, self).__init__(ctx)

    def _args_to_tuple(self, node):
        """Ties together all positional and *arg arguments
        in a single tuple."""
        builder = _ArgTemplateBuilder()
        for a in node.args:
            if isinstance(a, gast.Starred):
                builder.add_stararg(a.value)
            else:
                builder.add_arg(a)
        builder.finalize()
        return builder.to_ast()

    def _kwargs_to_dict(self, node):
        """Ties together all keyword and **kwarg arguments
        in a single dict."""
        if node.keywords:
            return gast.Call(
                gast.Name(
                    "dict",
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None,
                ),
                args=(),
                keywords=node.keywords,
            )
        else:
            return parser.parse_expression("None")

    def visit_Lambda(self, node):
        if not anno.hasanno(node, "function_context_name"):
            return self.generic_visit(node)
        with self.state[_Function] as fn_scope:
            fn_scope.context_name = anno.getanno(node, "function_context_name")
            return self.generic_visit(node)

    def visit_Call(self, node):
        full_name = str(anno.getanno(node.func, anno.Basic.QN, default=""))
        function_context_name = self.state[_Function].context_name
        full_bound_name = (
            node.func.attr if isinstance(node.func, gast.Attribute) else full_name
        )

        node = self.generic_visit(node)

        # Calls to the function context manager (inserted by function_scopes) are
        # also safe.
        if full_name.startswith(function_context_name + "."):
            return node

        # Avoid converting an already converted node call
        if full_name.__contains__("ivy__."):
            return node

        # Calls to pdb.set_trace or ipdb.set_trace are never converted. We don't use
        # the normal mechanisms to bypass these literals because they are sensitive
        # to the frame they are being called from.
        if full_name in ("pdb.set_trace", "ipdb.set_trace", "breakpoint"):
            global _SET_TRACE_WARNED
            if not _SET_TRACE_WARNED:
                _SET_TRACE_WARNED = True
            return node

        if (
            hasattr(node.func, "id")
            and node.func.id == "isinstance"
            and isinstance(node.args[1], gast.Attribute)
            and node.args[1].attr == "Size"
        ):
            new_call = gast.Call(
                func=gast.Name(
                    id="isinstance", ctx=gast.Load(), annotation=None, type_comment=None
                ),
                args=[
                    node.args[0],
                    gast.Name(
                        id="TrackedTupleProxy",
                        ctx=gast.Load(),
                        annotation=None,
                        type_comment=None,
                    ),
                ],
                keywords=[],
            )
            return new_call

        # convert Tensor() --> torch.tensor() as we cannot trace torch.Tensor.__init__
        if isinstance(node.func, gast.Name) and node.func.id == "Tensor":
            # Rename `Tensor` to `torch.tensor`
            node.func = gast.Attribute(
                value=gast.Name(
                    id="torch", ctx=gast.Load(), type_comment=None, annotation=None
                ),
                attr="tensor",
                ctx=gast.Load(),
            )
            return node
        elif (
            isinstance(node.func, gast.Attribute)
            and node.func.attr == "Tensor"
            and isinstance(node.func.value, gast.Name)
            and node.func.value.id == "torch"
        ):
            # Rename `torch.Tensor` to `torch.tensor`
            node.func.attr = "tensor"
            return node

        if any((full_name == x.__name__ for x in glob.BUILTIN_CALLABLES_TO_TRACK)):
            self.contains_builtins = (
                True if not self.contains_builtins else self.contains_builtins
            )

            # Transform the builtin function call
            wrapped_func_name = f"wrapped_builtin_callable"
            wrapped_func = gast.Name(
                id=wrapped_func_name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None,
            )
            new_args = [
                gast.Name(
                    id=node.func.id, ctx=gast.Load(), annotation=None, type_comment=None
                )
            ] + node.args
            new_node = gast.Call(
                func=wrapped_func, args=new_args, keywords=node.keywords
            )

            # Copy linenos and other attrs to transformed node
            return gast.copy_location(new_node, node)
        else:
            if (
                self._transpiling
                and full_bound_name not in glob.BUILTIN_METHODS_TO_TRACK
            ):
                # No need to transform anything when doing the second compilation pass
                # during transpilation
                return node
            # Convert the call to lazily transform it whenever it is eagerly executed
            # down the call stack
            self.contains_lazy_transforms = (
                True
                if not self.contains_lazy_transforms
                else self.contains_lazy_transforms
            )
            template = """
                ivy__.converted_call(func, args, kwargs, function_ctx)
            """
            new_call = templates.replace_as_expression(
                template,
                func=node.func,
                args=self._args_to_tuple(node),
                kwargs=self._kwargs_to_dict(node),
                function_ctx=function_context_name,
            )
            return new_call

    def visit_FunctionDef(self, node):
        if node.body:
            # Strip the docstring node if it exsits
            first_statement = node.body[0]
            first_value = (
                first_statement.value if hasattr(first_statement, "value") else None
            )
            docstring_node = (
                first_statement
                if (
                    isinstance(first_statement, gast.Expr)
                    and first_value
                    and isinstance(first_value, gast.Constant)
                    and hasattr(first_value, "value")
                    and isinstance(first_value.value, str)
                )
                else None
            )

            # Insert the import statement
            import_statement = gast.ImportFrom(
                module="tracer.helpers",
                names=[
                    gast.alias(name="wrapped_builtin_callable", asname=None),
                ],
                level=0,
            )
            (
                node.body.insert(1, import_statement)
                if docstring_node is not None
                else node.body.insert(0, import_statement)
            )
            import_statement = gast.ImportFrom(
                module="tracer.tracked_var_proxy",
                names=[
                    gast.alias(name="TrackedTupleProxy", asname=None),
                ],
                level=0,
            )
            (
                node.body.insert(1, import_statement)
                if docstring_node is not None
                else node.body.insert(0, import_statement)
            )

        # Decorators and arg defaults are part of the outer scope.
        node.decorator_list = self.visit_block(node.decorator_list)
        node.args.defaults = self.visit_block(node.args.defaults)
        for i, d in enumerate(node.args.kw_defaults):
            if d is not None:
                node.args.kw_defaults[i] = self.visit(d)

        with self.state[_Function] as fn_scope:
            fn_scope.context_name = anno.getanno(node, "function_context_name")
            node = self.generic_visit(node)
            return node

    def visit_With(self, node):
        # Context manager calls (in node.items) are not converted.
        node.body = self.visit_block(node.body)
        return node


# BuiltinsTranspiler #
# ------------------ #


class BuiltinsTranspiler(py_transpiler.PyToPy):
    """Transpiler to convert given function and return the transformed function."""

    def __init__(self, obj, transpiling):
        super(BuiltinsTranspiler, self).__init__()
        self._obj = obj
        self.filename = ""
        self._transpiling = transpiling
        self._extra_locals = None
        self._cache_lock = threading.RLock()

    def _is_cached(self, fn, cache_subkey):
        global _TRANSFORMED_FN_CACHE
        try:
            return _TRANSFORMED_FN_CACHE.has(fn, cache_subkey)
        except TypeError:
            # Catch-all for entities that are unhashable or don't allow weakrefs.
            return False

    def _cached_factory(self, fn, cache_subkey):
        global _TRANSFORMED_FN_CACHE
        cached_factory = _TRANSFORMED_FN_CACHE[fn][cache_subkey]
        return cached_factory

    def _cache_factory(self, fn, cache_subkey, factory):
        global _TRANSFORMED_FN_CACHE
        if len(_TRANSFORMED_FN_CACHE) == _MAX_CACHE_SIZE:
            _ = _TRANSFORMED_FN_CACHE._cache.popitem()
        try:
            _TRANSFORMED_FN_CACHE[fn][cache_subkey] = factory
        except TypeError:
            # Catch-all for entities that are unhashable or don't allow weakrefs.
            pass

    def get_transformed_name(self, node):
        return "ivy__" + super(BuiltinsTranspiler, self).get_transformed_name(node)

    def get_caching_key(self, ctx):
        return ctx.options

    def get_extra_locals(self):
        if self._extra_locals is None:
            module_spec = importlib.machinery.ModuleSpec("autograph_ivy", None)
            ag_internal = importlib.util.module_from_spec(module_spec)
            ag_internal.__dict__.update(inspect.getmodule(BuiltinsTranspiler).__dict__)
            ag_internal.ConversionOptions = converter.ConversionOptions
            ag_internal.STD = converter.STANDARD_OPTIONS
            ag_internal.Feature = converter.Feature
            ag_internal.FunctionScope = function_wrappers.FunctionScope
            ag_internal.with_function_scope = function_wrappers.with_function_scope
            ag_internal.__dict__.update(operators.__dict__)
            self._extra_locals = {"ivy__": ag_internal}
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
        ctx.contains_lazy_builtins = False
        transformer = BuiltinCallsTransformer(
            self._obj.__name__, ctx, self._transpiling
        )
        node = transformer.visit(node)
        ctx.contains_lazy_builtins = (
            True
            if transformer.contains_builtins or transformer.contains_lazy_transforms
            else False
        )
        return node

    def transform_function(self, fn, user_context=None):
        cache_subkey = self.get_caching_key(user_context)
        if self._is_cached(fn, cache_subkey):
            # Check if the fn factory can be retrieved from an existing cache
            factory = self._cached_factory(fn, cache_subkey)
        else:
            with self._cache_lock:
                # Check again under lock.
                if self._is_cached(fn, cache_subkey):
                    factory = self._cached_factory(fn, cache_subkey)
                else:
                    nodes, ctx = super(py_transpiler.PyToPy, self).transform_function(
                        fn, user_context
                    )

                    # No transformation was performed, return the original fn
                    if not ctx.contains_lazy_builtins:
                        return fn, None, {}

                    if isinstance(nodes, gast.Lambda):
                        nodes = gast.Assign(
                            targets=[
                                gast.Name(
                                    ctx.info.name,
                                    ctx=gast.Store(),
                                    annotation=None,
                                    type_comment=None,
                                ),
                            ],
                            value=nodes,
                        )
                    else:
                        nodes.name = ctx.info.name

                    # Instantiate a new function factory for the transformed fn
                    factory = py_transpiler._PythonFnFactory(
                        ctx.info.name, fn.__code__.co_freevars, self.get_extra_locals()
                    )

                    # Use the instantiated factory to create a factory for the transformed fn
                    self.filename = factory.create(
                        nodes,
                        ctx.namer,
                        future_features=ctx.info.future_features,
                        return_filename=True,
                    )

                    # Cache the transformed fn factory to avoid re-transforming it downstream
                    self._cache_factory(fn, cache_subkey, factory)

        # Instantiate the transformed fn using the factory
        transformed_fn = factory.instantiate(
            globals_=fn.__globals__,
            closure=fn.__closure__ or (),
            defaults=fn.__defaults__,
            kwdefaults=getattr(fn, "__kwdefaults__", None),
        )

        # Mark as transformed
        _setattr_transformed(transformed_fn)

        # Save a reference of the original non-transformed object
        setattr(transformed_fn, "__wrapped__", self._obj)

        return transformed_fn, factory.module, factory.source_map

    def transform(self, user_context=None):
        """Transforms a Python object."""
        if (
            inspect.isfunction(self._obj)
            or inspect.ismethod(self._obj)
            or isinstance(self._obj, str)
        ):
            return self.transform_function(self._obj, user_context)
        raise NotImplementedError("Non-function: {}".format(type(self._obj)))


# Interface #
# ---------- #


def transform_builtins(
    obj: Any, user_context=None, /, *, depth=None, transpiling=False
) -> Any:
    """Applies AST Transformation to entity."""
    from tracer.graph import Graph, LazyGraph

    depth = 1 if transpiling else depth
    global _FUNCTIONLIKE, _BUILTIN_FUNCTIONLIKE

    # If we are at depth == 0, we need to reset the cache here
    if depth == 0:
        global _TRANSFORMED_FN_CACHE
        _TRANSFORMED_FN_CACHE = cache.CodeObjectCache()

    is_graph_obj = isinstance(obj, Graph)
    is_lazy_graph_obj = isinstance(obj, LazyGraph)
    is_ivy_module_call = _is_ivy_module_call(obj)
    is_unsupported_obj = _is_unsupported(obj)
    has_code_obj = _exposes_code(obj)
    is_callable_func = _is_callable_funclike(obj)
    is_callable_obj = (
        _is_callable_funclike(obj)
        and not isinstance(obj, _FUNCTIONLIKE)
        and not is_graph_obj
    )

    if (not is_graph_obj) and (
        is_lazy_graph_obj
        or (is_unsupported_obj and not is_ivy_module_call)
        or (not is_callable_func and not has_code_obj)
        or (
            isinstance(obj, _BUILTIN_FUNCTIONLIKE)
            and obj not in glob.BUILTIN_CALLABLES_TO_TRACK
        )
    ):
        return obj

    # Callable objects
    if is_callable_obj or is_ivy_module_call:
        # If we have a ivy.Module._call here, we need to transform the
        # underlying native_module here
        base_ivy_obj = None
        if is_ivy_module_call:
            base_ivy_obj = obj
            obj = obj.__self__._native_module

        # Otherwise, get the effective callable object that needs to be transformed.
        # The obj could either be a trainable module like torch.nn.LSTM instance or it can be
        # its bound __call__ method of that instance
        is_callable_instance = hasattr(obj, "__class__") and _is_callable_funclike(
            obj.__class__
        )
        if is_callable_instance:
            base_obj = obj.__class__
            callable_obj = obj.__class__.__call__
        else:
            base_obj = obj
            callable_obj = obj.__call__

        # Transform the callable object
        if callable_obj is not base_obj:
            try:
                transformed = transform_builtins(callable_obj, user_context)
                setattr(base_obj, "__call__", transformed)
                glob.transformed_callables.append((base_obj, transformed))
            except:
                pass

        # Restore the transformed object correctly
        if is_ivy_module_call:
            if is_callable_instance:
                base_ivy_obj.__self__._native_module.__class__ = base_obj
            else:
                base_ivy_obj.__self__._native_module = base_obj
            return base_ivy_obj

        return obj

    # If a Graph object is passed, transform its scripted call
    callable_obj = obj._scripted_call if is_graph_obj else obj

    # Check if it is transformed already and return early
    if _is_transformed(callable_obj):
        return callable_obj

    # Transform the given function
    try:
        if user_context is None:
            user_context = converter.ProgramContext(
                options=converter.ConversionOptions(user_requested=True)
            )
        builtins_transpiler = BuiltinsTranspiler(callable_obj, transpiling)
        transformed, module, source_map = builtins_transpiler.transform(
            user_context=user_context
        )
        if os.getenv("IVY_DEBUG_SOURCE", "False").lower() == "true":
            atexit.register(lambda: loader._remove_file(builtins_transpiler.filename))
        else:
            loader._remove_file(builtins_transpiler.filename)
    except Exception as _:
        transformed, module, source_map = callable_obj, None, {}

    # No transformation was performed, return the original entity
    if transformed is callable_obj:
        if is_graph_obj or is_ivy_module_call:
            return obj
        else:
            return callable_obj

    # Set attributes if input was a Graph instance
    if is_graph_obj:
        obj._scripted_call = transformed
        obj._scripted_call.module = module
        obj._scripted_call.source_map = source_map
        return obj

    transformed.module = module
    transformed.source_map = source_map

    # At depth == 0, need to take care of instance methods by
    # isolating the self reference before the call
    if depth == 0:
        if hasattr(obj, "__self__") and not hasattr(transformed, "__self__"):
            original = transformed

            @functools.wraps(original)
            def wrapped_transformed(*args, **kwargs):
                return original(obj.__self__, *args, **kwargs)

            transformed = wrapped_transformed
    return transformed
