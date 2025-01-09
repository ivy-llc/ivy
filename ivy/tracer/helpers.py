# global
import collections
import copy
import enum
import functools
import inspect
import os
from pathlib import Path
import threading
from types import FunctionType, ModuleType, MethodType, BuiltinMethodType
from typing import Callable
import weakref

# local
import ivy
from . import globals as glob
from .conversion import is_array, array_to_new_backend, nest_array_to_new_backend
from .numpy_proxy import custom_np_classes
from .tracked_var_proxy import TrackedVarProxy


class NoParametersError(Exception):
    pass


def _is_submodule(obj, kw):
    cls_str = {
        "torch": "torch.nn.modules.module.Module",
        "keras": "keras.engine.training.Model",
        "haiku": "haiku._src.transform.Transformed",
        "paddle": "paddle.fluid.dygraph.layers.Layer",
        "flax": "flax.linen.module.Module",
    }[kw]
    try:
        for bc in type(obj).mro():
            if cls_str in str(bc):
                return True
    except TypeError:
        pass
    return False


def _check_is_trainable_module(obj, to):
    is_trainable_module = False
    to_mod = None

    if to == "torch" or _is_submodule(obj, "torch"):
        import torch

        if isinstance(obj, torch.nn.Module):
            is_trainable_module = True
            to = "torch"
            to_mod = "torch"

    elif to == "tensorflow" or _is_submodule(obj, "keras"):
        import tensorflow as tf

        if isinstance(obj, tf.keras.Model):
            is_trainable_module = True
            to = "tensorflow"
            to_mod = "keras"

    elif to == "paddle" or _is_submodule(obj, "paddle"):
        import paddle

        if isinstance(obj, paddle.nn.Layer):
            is_trainable_module = True
            to = "paddle"
            to_mod = "paddle"

    elif to == "jax" or _is_submodule(obj, "haiku") or _is_submodule(obj, "flax"):
        if _is_submodule(obj, "haiku"):
            import haiku as hk

            if isinstance(obj, hk.Transformed):
                is_trainable_module = True
                to = "jax"
                to_mod = "haiku"

        elif _is_submodule(obj, "flax"):
            import flax

            if isinstance(obj, flax.linen.Module):
                is_trainable_module = True
                to = "jax"
                to_mod = "flax"

    # check if obj is an ivy.Module instance
    if isinstance(obj, ivy.Module):
        to = "ivy"
        is_trainable_module = True
        to_mod = "ivy"

    return is_trainable_module, to, to_mod


def _get_arg_callback_functions(args):
    """
    Retrieve any callback functions within the given arguments, along with the
    index representing the position within the arguments of that function.

    Returns:
        List of (idx, function) pairs
    """
    from tracer.graph import LazyGraph

    callback_functions = []
    idxs = []

    for idx, arg in enumerate(args):
        if callable(arg) and not isinstance(arg, LazyGraph):
            callback_functions.append(arg)
            idxs.append(idx)

    return zip(idxs, callback_functions)


def _get_kwarg_callback_functions(kwargs):
    """
    Retrieve any callback functions within the given key-word arguments, along with
    the key that can be used to retrieve that function from the kwargs dict.

    Returns:
        List of (key, function) pairs
    """
    from tracer.graph import LazyGraph

    callback_functions = []
    keys = []

    for key, value in kwargs.items():
        if callable(value) and not isinstance(value, LazyGraph):
            callback_functions.append(value)
            keys.append(key)

    return zip(keys, callback_functions)


def _get_ivy_key() -> str:
    """
    Attempts to retrieve a locally stored ivy api key, either from the IVY_KEY
    environment variable, or by recursively searching for a ivy-key.pem file.
    """
    try:
        if "IVY_KEY" not in os.environ:
            # traverse backwards through the directory tree, searching for ivy-key.pem
            current_dir = os.getcwd()
            key_path = None

            while current_dir != os.path.dirname(current_dir):  # stop at root directory
                possible_key_path = os.path.join(current_dir, "ivy-key.pem")
                possible_ivy_root = os.path.join(current_dir, ".ivy")
                if os.path.isfile(possible_key_path):
                    key_path = possible_key_path
                    break
                elif os.path.isdir(possible_ivy_root):
                    possible_key_path = os.path.join(possible_ivy_root, "ivy-key.pem")
                    if os.path.isfile(possible_key_path):
                        key_path = possible_key_path
                        break
                current_dir = os.path.dirname(current_dir)
                
            with open(key_path, "r") as key_file_py:
                key = key_file_py.readline()
        else:
            key = os.environ.get("IVY_KEY")
        return key if key != "" else None
    except:
        return None


def _convert_callbacks_to_lazy_graphs(graph, fn, args, kwargs):
    """
    Convert the callback function(s) present in the args into LazyGraphs so
    they will be traced and converted to graphs when the function is run
    """
    from tracer.graph import LazyGraph
    from tracer.tracer import trace_subgraph

    subgraphs = []

    new_args = list(args)
    new_kwargs = dict(kwargs)

    arg_callback_fns = _get_arg_callback_functions(args)
    kwarg_callback_fns = _get_kwarg_callback_functions(kwargs)

    # wrap any callback functions found in the args
    for idx, callback_fn in arg_callback_fns:
        # replace function with LazyGraph
        callback_lazy_graph = LazyGraph(
            callback_fn,
            initializer=trace_subgraph,
            id_to_fn=graph._id_to_function,
            id_to_param=graph._id_to_parameter,
        )
        new_args[idx] = callback_lazy_graph
        subgraphs.append((idx, callback_lazy_graph))

    # wrap any callback functions found in the kwargs
    for key, callback_fn in kwarg_callback_fns:
        # replace function with LazyGraph
        callback_lazy_graph = LazyGraph(
            callback_fn,
            initializer=trace_subgraph,
            id_to_fn=graph._id_to_function,
            id_to_param=graph._id_to_parameter,
        )
        new_kwargs[key] = callback_lazy_graph
        subgraphs.append((key, callback_lazy_graph))

    return subgraphs, tuple(new_args), dict(new_kwargs)


def _deepcopy(x, memo=None, _nil=[]):
    if isinstance(x, ivy.Container):
        return x.cont_deep_copy()
    trace_classes = glob.trace_classes
    glob.trace_classes = False
    ret = copy.deepcopy(x, memo=memo, _nil=_nil)
    glob.trace_classes = trace_classes
    return ret


def _copy(x):
    return copy.copy(x)


def _find_missing_frontends(graph):
    def in_frontend(path):
        # Check if the path belongs to a built-in module
        for module in glob.BUILTIN_MODULES_TO_TRACK:
            if path.startswith(module + "."):
                return True

        path_to_replace = glob.NATIVE_TO_FRONTEND_PATH.keys()
        if any([p in path for p in path_to_replace]):
            return True
        frontend_path = "ivy.functional.frontends." + path
        try:
            exec(frontend_path)
        except AttributeError:
            return False
        return True

    to_ignore = ["__getattribute__", "__getattr__", "__getitem__"]
    backend_paths = [f.path for f in graph._functions if hasattr(f, "path")]
    missing_paths = [p for p in backend_paths if not in_frontend(p)]
    missing_paths = [mp for mp in missing_paths if mp.split(".")[-1] not in to_ignore]
    # get an ordered counter with (fn_path_str: number_of_occurrences)
    frequency = collections.Counter(missing_paths).most_common()
    return frequency


def _format_missing_frontends_msg(frequency):
    msg = (
        "There are functions that are not yet implemented in the Ivy frontend API. "
        + "Visit Ivy's open task page to learn more about contributing to the frontend APIs! "
        + "https://lets-unify.ai/ivy/contributing/open_tasks.html\n"
        + "The missing functions are <(number of calls) function_path> : \n-> {}".format(
            "\n-> ".join(
                [" (" + str(freq[1]) + ") \t" + str(freq[0]) for freq in frequency]
            )
        )
    )
    if not frequency:  # No missing frontends
        msg = "All the functions in this graph are implemented in the Ivy frontend API!"
    return msg


def _is_untrackable(var, with_numpy=True, stateful_classes=()) -> bool:
    """Checks if a given variable is an instance of a non-array class to track by checking
    whether it contains other wrapped classes nested inside or not. If it does, we will not
    track it with our proxies e.g. Sequence[Union[torch.Tensor, tf.EagerTensor, ...]]
    Parameters
    ----------
    var
        Variable to check.
    with_numpy
        Whether we are tracing the graph with numpy
    stateful_classes
        Classes to be considered stateful during tracing

    Returns
    -------
        True if the variable cannot be tracked by our proxies, False otherwise.
    """
    # If any of the nests contain instances of logged classes i.e. Tensors, Arrays etc,
    # the nest cannot be tracked by our proxies.
    if isinstance(var, (tuple, list, dict)):
        return ivy.nested_any(
            var,
            lambda x: _is_untrackable(
                x, with_numpy=with_numpy, stateful_classes=stateful_classes
            ),
        )
    return is_array(var, with_numpy=with_numpy) or isinstance(var, stateful_classes)


def _is_untracked_enum(var) -> bool:
    """Checks if a given variable is an enum instance that should be tracked."""
    return isinstance(var, (enum.Enum, enum.IntEnum)) and "get_var" not in dir(var)


def _is_tracked_variable(var) -> bool:
    """Checks if var is a tracked variable proxy."""
    return isinstance(var, TrackedVarProxy)


def _is_tracked_np_proxy(var) -> bool:
    """Checks if var is a tracked numpy proxy."""
    return var.__class__ in custom_np_classes


def contain_stored_class_instance(args) -> bool:
    """Checks if the args contain the class instance as the first argument."""
    return id(type(args[0])) in glob.class_instances


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def _replace_qualname(obj, old_string, new_string):
    if hasattr(obj, "__qualname__") and obj.__qualname__:
        try:
            obj.__qualname__ = obj.__qualname__.replace(old_string, new_string)
        except:
            try:
                setattr(
                    obj,
                    "__qualname__",
                    obj.__qualname__.replace(old_string, new_string),
                )
            except Exception as _:
                pass
        return obj
    else:
        return obj


def _copy_cls(cls):
    # Copy over the functions first
    functions = inspect.getmembers(
        cls, predicate=lambda member: inspect.isfunction(member)
    )

    members = []
    new_dict = {}
    cls_name = cls.__name__
    for funcname, func in functions:
        try:
            func = copy.deepcopy(func)
        except:
            pass

        new_dict[funcname] = func
        members.append((funcname, func))

    # Create a new class
    try:
        copied_cls = type(cls)(f"{cls_name}", (cls,), new_dict)
    except Exception as e:
        try:
            # alternative method, subclassing
            class CopiedClass(cls):
                pass

        except:
            # the above fails for classes that can't be inherited.
            class CopiedClass:
                pass

        copied_cls = CopiedClass
        for funcname in new_dict.keys():
            type.__setattr__(copied_cls, funcname, new_dict[funcname])
        copied_cls.__name__ = cls_name

    # Now copy over all the methods
    methods = inspect.getmembers(cls, predicate=lambda member: inspect.ismethod(member))
    for methodname, method in methods:
        method = MethodType(method, copied_cls)
        setattr(copied_cls, methodname, method)
        members.append((methodname, method))

    # Next, copy over any class attributes
    class_attrs = [
        attr for attr in inspect.classify_class_attrs(cls) if attr.kind == "data"
    ]
    class_attrs = [attr for attr in class_attrs if not attr.name.startswith("__")]
    return copied_cls, members, class_attrs


def _apply_fn_to_class(cls, fn, fargs, fkwargs, **kwargs):
    """
    Applies a function to all methods in a given class.
    Avoids modifying the original class.
    """

    def decorator(fn, fargs, fkwargs, **kwargs):
        def decorated(inner_function):
            try:
                name = inner_function.__name__
            except AttributeError:
                name = "function"

            # Only the constructor is supposed to recieve the args/kwargs
            # when tracing/transpiling entire classes
            if name != "__init__":
                return MethodDescriptor(inner_function, name, fn, None, None, **kwargs)
            return MethodDescriptor(inner_function, name, fn, fargs, fkwargs, **kwargs)

        return decorated

    copied_cls, methods, class_attrs = _copy_cls(cls)

    class_dict = {**cls.__dict__}

    for base in inspect.getmro(cls):
        class_dict = {**class_dict, **(base.__dict__)}

    # Methods
    for membername, method in methods:
        wrap_fn = method

        if isinstance(class_dict[membername], staticmethod):
            # Static methods
            wrap_fn = fn(
                staticmethod(method),
                **kwargs,
            )
        elif isinstance(class_dict[membername], classmethod):
            # Class methods
            method = method.__func__
            wrap_fn = fn(classmethod(method).__func__, **kwargs)
        else:
            # Instance methods
            wrap_fn = decorator(fn, fargs=fargs, fkwargs=fkwargs, **kwargs)(method)

        setattr(copied_cls, membername, wrap_fn)

    # Class attributes
    transpiling = kwargs.get("source", False)
    if transpiling:
        ivy.set_backend(kwargs["to"])
    for attr in class_attrs:
        try:
            if transpiling:
                from tracer.conversion import nest_array_to_new_backend

                setattr(copied_cls, attr.name, nest_array_to_new_backend(attr.object))
            else:
                setattr(copied_cls, attr.name, attr.object)
        except:
            setattr(copied_cls, attr.name, attr.object)
    if transpiling:
        ivy.previous_backend()

    return copied_cls


def _copy_module(module):
    ret = ModuleType(module.__name__, module.__doc__)
    ret.__class__ = type(module)
    ret.__dict__.update(module.__dict__)
    return ret


def _apply_fn_to_module(module, fn, *args, visited=None, **kwargs):
    """
    Applies a function to all methods in a given module.
    Avoids modifying the original module.
    """
    module = _copy_module(module)
    name = "/".join(module.__name__.split("."))
    members = dir(module)
    visited = ivy.default(visited, dict())
    for m in members:
        val = getattr(module, m)

        # do not apply transpilation to any unnecessary members
        if fn.__name__ == "transpile":
            if hasattr(val, "__module__") and module.__name__ not in val.__module__:
                continue

        if (
            isinstance(val, ModuleType)
            and "__file__" in val.__dict__
            and name in val.__file__
        ):
            if val not in visited:
                visited[val] = True
                setattr(
                    module,
                    m,
                    _apply_fn_to_module(
                        val,
                        fn,
                        *args,
                        visited=visited,
                        **kwargs,
                    ),
                )
        elif isinstance(val, Callable):
            setattr(module, m, fn(val, *args, **kwargs))
        elif isinstance(val, type):
            setattr(module, m, fn(val, *args, **kwargs))

    return module


def _give_same_argspec(fn, fn2, kwonly=False):
    """
    Creates a wrapper for fn2 which looks identical to fn,
    including its FullArgSpec.
    """
    try:
        spec = inspect.getfullargspec(fn)
    except:
        return functools.wraps(fn)(fn2)

    namespace = {
        "fn": fn,
        "fn2": fn2,
        "functools": functools,
        "spec": spec,  # for debugging
    }

    def to_str(obj):
        nonlocal namespace
        namespace["def" + str(id(obj))] = obj
        return "=def" + str(id(obj))

    def get_ann(arg):
        nonlocal namespace
        if arg not in spec.annotations:
            return ""
        else:
            cls = spec.annotations[arg]
            namespace["cls" + str(id(cls))] = cls
            return ":cls" + str(id(cls))

    def get_kw_default(kw):
        nonlocal namespace
        if not kw in spec.kwonlydefaults:
            return ""
        default = spec.kwonlydefaults[kw]

        # preserve reference after pickling
        if default.__class__ is object:
            default = weakref.ref(default)

        namespace["def" + str(id(default))] = default
        return "=def" + str(id(default))

    argstring = ""
    posargs_string = ""
    callstring = ""
    posargs_callstring = ""

    reversed_defaults = []
    defaults = spec.defaults
    if not spec.defaults:
        defaults = []
    args = copy.copy(spec.args)
    if not spec.args:
        args = []
    kwonlyargs = spec.kwonlyargs
    if not spec.kwonlyargs:
        kwonlyargs = []
    for default in defaults:
        reversed_defaults.insert(0, default)
    for default in reversed_defaults:
        arg = args[-1]
        argstring = arg + get_ann(arg) + to_str(default) + "," + argstring
        if len(kwonlyargs) > 0 and not arg in kwonlyargs:
            callstring = arg + "," + callstring
        else:
            callstring = arg + "=" + arg + "," + callstring
        args.pop()
    for arg in args:
        posargs_string += arg + ","
        if not kwonly:
            posargs_callstring += arg + ","
        else:
            posargs_callstring += arg + "=" + arg + ","

    if spec.varargs:
        argstring += "*" + spec.varargs + get_ann(spec.varargs) + ","
        callstring += "*" + spec.varargs + ","
    for kwonlyarg in kwonlyargs:
        argstring += kwonlyarg + get_ann(kwonlyarg) + get_kw_default(kwonlyarg) + ","
        callstring += kwonlyarg + "=" + kwonlyarg + ","
    if spec.varkw:
        argstring += "**" + spec.varkw + get_ann(spec.varkw) + ","
        callstring += "**" + spec.varkw + ","

    sigstring = posargs_string + argstring
    callstring = posargs_callstring + callstring

    return_annotation = ""
    if "return" in spec.annotations:
        namespace["typ" + str(id(spec.annotations["return"]))] = spec.annotations[
            "return"
        ]
        return_annotation = "->" + "typ" + str(id(spec.annotations["return"]))

    code = """
@functools.wraps(fn)
def new_fn({}){}:
    return fn2({})
    """.format(
        sigstring, return_annotation, callstring
    )

    # for debugging.
    namespace["code"] = code

    exec(code, namespace)

    namespace["new_fn"].__wrapped__ = fn
    return namespace["new_fn"]


def _simple_wrap(fn, fn2):
    """
    Wraps fn with fn2 in such a manner that it can be correctly
    unwrapped when the tracer-transpiler is cythonized.
    """

    namespace = {
        "fn": fn,
        "fn2": fn2,
        "functools": functools,
    }

    code = """
@functools.wraps(fn)
def new_fn(*args, **kwargs):
    return fn2(*args, **kwargs)
    """

    # for debugging.
    namespace["code"] = code

    exec(code, namespace)

    namespace["new_fn"].__wrapped__ = fn
    return namespace["new_fn"]


def class_method_to_instance_method(method_wrapper, instance):
    """Constructs a new `MethodDescriptor` with `self` bound."""
    # weak_instance = weakref.ref(instance)

    # While we could bind to a weakref proxy instead, that causes the
    # bound method to be unhashable.
    bound_method = MethodType(
        method_wrapper.method, MethodTarget(instance, method_wrapper.method)
    )

    def bound_method_wrapper(*args, **kwargs):
        """Wraps a dummy MethodType."""
        wrapped_fn = method_wrapper.method
        return wrapped_fn(instance, *args, **kwargs)

    # We make a dummy MethodType object to generate the correct bound method
    # signature. The actual call is to a function with a weak reference to
    # `instance`.
    instance_func = type(method_wrapper)(
        _give_same_argspec(bound_method, bound_method_wrapper),
        method_wrapper.name,
        method_wrapper._fn,
        method_wrapper._args,
        method_wrapper._kwargs,
        **method_wrapper._fn_kwargs,
    )
    return instance_func


class MethodDescriptor:
    def __init__(self, method, name, fn_to_apply, fargs, fkwargs, **kwargs):
        self._lock = threading.RLock()
        self._method = method
        self._name = name
        self._fn = fn_to_apply
        self._args = fargs
        self._kwargs = fkwargs
        self._fn_kwargs = kwargs
        self._converted = False
        self._graph = None
        self._descriptor_cache = weakref.WeakKeyDictionary()

    @property
    def name(self):
        return self._name

    @property
    def method(self):
        return self._method

    @property
    def graph(self):
        return self._graph

    def __get__(self, instance, owner):
        del owner

        # Need to bound instance methods to self here
        return self.get_bounded_instance(instance)

    def __call__(self, *args, **kwargs):
        with self._lock:
            if hasattr(self.method, "__wrapped__"):
                f = self.method.__wrapped__
            else:
                f = self.method
            f_self = f.__self__

            if inspect.ismethod(f) and isinstance(f_self, MethodTarget):
                fn = f_self.weakrefself_func__()
                args = (f_self.weakrefself_target__,) + args

                # TODO: Correct the arg_stateful_idxs order in case
                # the user also passes stateful args, instead of hardcoding
                # to [[0]]
                self._fn_kwargs["arg_stateful_idxs"] = ivy.default(
                    self._fn_kwargs.get("arg_stateful_idxs"), []
                )

                (
                    self._fn_kwargs["arg_stateful_idxs"].append([0])
                    if [0] not in self._fn_kwargs["arg_stateful_idxs"]
                    else self._fn_kwargs["arg_stateful_idxs"]
                )
            else:
                if hasattr(self.method, "__wrapped__"):
                    fn = self.method.__wrapped__
                else:
                    fn = self.method

            # this method is already transpiled
            if self._converted:
                return self._graph(*args, **kwargs)

            # we don't trace these as this leads to unbounded recursion (example: use of __setstate__ in copy.deepcopy).
            if hasattr(fn, "__name__") and fn.__name__ in [
                "__getattr__",
                "__getattribute__",
                "__getstate__",
                "__setstate__",
                "__setattr__",
            ]:
                return fn(*args, **kwargs)

            # we should return the original method if we are
            # in the middle of transpiling another method
            # otherwise we start to trace graph compiler code.
            if glob.tracing_paused == False:
                return fn(*args, **kwargs)

            if glob.trace_classes == False:
                return fn(*args, **kwargs)

            def _is_init():
                return hasattr(fn, "__name__") and fn.__name__ == "__init__"

            def _is_transpile():
                return hasattr(self._fn, "name") and self._fn.__name__ == "transpile"

            if _is_transpile():
                source = self._fn_kwargs["source"]
                to = self._fn_kwargs["to"]

            def use_source_init(args, kwargs):
                if _is_init() and self._fn.__name__ == "transpile":

                    # init function that uses the original backend
                    def source_init(self, *args, **kwargs):
                        glob.tracing_paused = False

                        ivy.set_backend(source)
                        args = nest_array_to_new_backend(args)
                        kwargs = nest_array_to_new_backend(kwargs)

                        fn(self, *args, **kwargs)

                        # save instance
                        glob.class_instances[id(type(self))] = copy.deepcopy(self)

                        is_array = {}
                        for k in object.__dir__(self):
                            attr = getattr(self, k)
                            is_array[k] = ivy.is_array(attr)
                        ivy.set_backend(to)

                        for k in object.__dir__(self):
                            if is_array[k]:
                                attr = getattr(self, k)
                                try:
                                    object.__setattr__(
                                        self, k, array_to_new_backend(attr)
                                    )
                                except AttributeError:
                                    pass

                        glob.tracing_paused = True

                    self._converted = True
                    self._graph = source_init
                    return source_init(*args, **kwargs)

            # transpilation / compilation step
            try:
                if len(args) > 0:
                    base_dir = object.__dir__(args[0])

                self._graph = self._fn(fn, args=args, kwargs=kwargs, **self._fn_kwargs)
                self._converted = True

                args = nest_array_to_new_backend(args)

                return self._graph(*args, **kwargs)

            except NoParametersError as e:
                if len(args) == 1:
                    return fn(*args, **kwargs)
                elif _is_init():
                    raise e
                    return use_source_init(args, kwargs)
                raise e

    def get_bounded_instance(self, instance):
        if instance not in self._descriptor_cache:
            if instance is None:
                return self
        else:
            return self._descriptor_cache[instance]

        bounded_instance = class_method_to_instance_method(self, instance)
        self._descriptor_cache[instance] = bounded_instance
        return bounded_instance


class MethodTarget:
    """When a method is bound to objects of this type,
    it allows us to recover a weak reference the
    original method's self pointer, so that we can execute
    it consistent with class_method_to_instance_method's
    bound_method_wrapper."""

    __slots__ = ("weakrefself_target__", "weakrefself_func__")

    def __init__(self, target, original_method):
        self.weakrefself_target__ = target
        self.weakrefself_func__ = weakref.ref(original_method)

    @property
    def target(self):
        return self.weakrefself_target__

    @property
    def target_class(self):
        true_self = self.weakrefself_target__
        if inspect.isclass(true_self):
            # Class method
            return true_self
        else:
            return true_self.__class__

    def call(self, args, kwargs):
        wrapped_fn = self.weakrefself_func__()
        return wrapped_fn(self.weakrefself_target__, *args, **kwargs)


def wrapped_builtin_callable(obj, *args, **kwargs):
    return obj(*args, **kwargs)


class WrappedCallable:
    def __init__(self, fn):
        self.__wrapped__ = fn.__wrapped__
        self.fn = fn
        self.__name__ = self.fn.__name__

    def __call__(self, *args, **kwargs):
        return self.fn.__call__(*args, **kwargs)

    def __reduce__(self):
        fn = self.fn
        if ivy.current_backend_str() == "torch":
            fn_qualname = getattr(self.fn, "__qualname__", "")
            if fn_qualname.startswith("_VariableFunctionsClass"):
                fn = self.fn.__wrapped__
        return (self.__class__, (fn,))


def _module_name(obj):
    try:
        ret = inspect.getmodule(obj)
    except:
        ret = ""

    if hasattr(ret, "__name__"):
        ret = ret.__name__

    if ret is None:
        ret = ""

    return ret


def _wraps(fn, from_tracked_var=False, is_builtin_fn=False):
    """
    Decorator that behaves like functools.wraps, but it mimics argspec and also
    adds the result to the global wrapped_fns dict.
    """

    def decorator(fn2):
        mod_name = _module_name(fn)
        if (
            isinstance(fn, (FunctionType))
            and not mod_name.startswith("tensorflow.python.ops")
            and not mod_name.startswith("jax._src.numpy")
        ):
            ret = _give_same_argspec(fn, fn2)
        else:
            ret = _simple_wrap(fn, fn2)
        if isinstance(fn, (BuiltinMethodType)):
            if not is_builtin_fn:
                ret = WrappedCallable(ret)
        if not from_tracked_var:
            glob.wrapped_fns[id(fn)] = (fn, ret)
        return ret

    return decorator


def copy_dict(x):
    if not isinstance(x, dict):
        return x
    y = {}
    for k, v in x.items():
        y[k] = v
    return y
