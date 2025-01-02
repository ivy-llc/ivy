# global
import ast
import inspect
from types import ModuleType, FunctionType, MethodType, BuiltinFunctionType
import traceback
import sys
import gc
import functools

# local
from . import globals as glob
from .helpers import WrappedCallable

# propagate changes to these modules
modules_to_update = [
    "torch",
    "tensorflow",
    "numpy",
    "jax",
    "ivy",
    "jaxlib",
    "keras",
    "paddle",
]

# track framework modules that are being uesd
visible_frameworks = {}

# does not attempt to reload these modules
modules_to_ignore = [
    "importlib",
    "inspect",
    "_pytest",
    "py",
    "tensorflow_probability",
    "tracer",
    "transpiler",
    "_dummy_thread",
    "typing",
    *sys.builtin_module_names,
]

MAX_OBJS = 100000

class_replacements = {}

function_replacements = {}

fw_function_replacements = {}

functions_to_ignore = {
    **{id(type.__getattribute__(object, k)): True for k in dir(object)},
    **{id(type.__getattribute__(dict, k)): True for k in dir(dict)},
}

searched_mutables = {}

classes_to_ignore = {}

reloaded = {}

is_relevant = {}

touched_module_ids = {}


class Namespace:
    def __init__(self, _dict):
        self.classes = {}
        self.callables = {}
        self.functions = {}
        for key in _dict:
            val = _dict[key]
            if hasattr(val, "__call__"):
                self.callables[key] = val
            if inspect.isclass(val):
                self.classes[key] = val
            if isinstance(val, FunctionType):
                self.functions[key] = val


def bind_method(fn, obj):
    return MethodType(fn, obj)


def patch_higher_modules():
    """
    Patch reloaded classes and functions which appear higher in the
    module dependency tree.
    """

    visited = {}
    memo = {}

    def has_touched_function_or_class(module):
        # todo: also look for class extensions of reloaded classes.
        classes = [
            safe_getattr(module, k)
            for k in ibdir(module)
            if isinstance(safe_getattr(module, k), type)
        ]
        functions = [
            safe_getattr(module, k)
            for k in ibdir(module)
            if isinstance(safe_getattr(module, k), FunctionType)
        ]

        for cls in classes:
            if id(cls) in class_replacements:
                return True
        for fn in functions:
            if id(fn) in function_replacements:
                return True
        return False

    def get_submodules(module):
        submodules = []
        for attr_name in ibdir(module):
            attr = safe_getattr(module, attr_name)
            if attr is None:
                continue
            if isinstance(attr, ModuleType):
                submodules.append(attr)
                continue
            if hasattr(attr, "__module__"):
                if attr.__module__ is module:
                    continue
                submodules.append(attr.__module__)
        return submodules

    # Simpler version of recursive reload.
    def rec_reload(module):
        nonlocal memo
        if not isinstance(module, ModuleType):
            return False
        if first_name(get_name(module)) in modules_to_ignore:
            return False
        else:
            pass
        if first_name(get_name(module)) in modules_to_update:
            return False
        if id(module) in touched_module_ids:
            return True
        if id(module) in memo:
            return memo[id(module)]
        submodules = get_submodules(module)
        reload_this = False
        if has_touched_function_or_class(module):
            reload_this = True
        visited[id(module)] = True
        for submodule in submodules:
            if id(submodule) in visited:
                continue
            elif rec_reload(submodule):
                reload_this = True

        memo[id(module)] = reload_this
        if reload_this:
            try:
                reload(module)
            except:
                print("failed to reload", module)
                print(traceback.format_exc())
                pass
            return True
        return False

    for module in sys.modules.copy().values():
        rec_reload(module)


def reset_reloader_globals():
    global reloaded, is_relevant, function_replacements, class_replacements, touched_module_ids
    global fw_function_replacements, classes_to_ignore, searched_mutables
    reloaded, is_relevant = {}, {}
    touched_module_ids = {}
    # may find a better way to clean up these dicts
    # perhaps using garbage collection
    class_replacements = {}
    function_replacements = {}
    fw_function_replacements = {}
    classes_to_ignore = {}
    searched_mutables = {}


def reset_globals_after(fn):
    def new_fn(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if glob.use_reloader:
            reset_reloader_globals()
        return ret

    return new_fn


@reset_globals_after
def apply_and_reload(to_reload, to_apply=lambda: None, args=[], kwargs={}, stateful=[]):
    """
    Calls to_apply with given arguments, then replaces all relevant references to framework
    functions and methods with their wrapped versions.

    Parameters
    ----------
    to_reload
        The callable to be reloaded, along with its dependencies.
    to_apply
        The callable to apply prior to reloading.
    args
        List of positional arguments for to_apply.
    kwargs
        Dict of keyword arguments for to_apply.
    stateful
        Classes that have already been wrapped that we shouldn't reload.

    Returns
    -------
    The reloaded callable.
    """
    if not glob.use_reloader:
        to_apply(*args, **kwargs)
        return to_reload

    global reloaded, is_relevant, function_replacements, class_replacements, touched_module_ids, classes_to_ignore
    global fw_function_replacements
    reset_reloader_globals()

    # currently, the tracer gets confused because we are replacing a stateful class.
    # since the tracer module is blacklisted from reloading,
    # the logged methods belong to the old one and aren't getting updated.
    # this line here is a quick fix but another solution should be found.

    classes_to_ignore = {k: True for k in stateful}

    for f in functions_to_ignore:
        classes_to_ignore.pop(id(f), None)

    to_apply(*args, **kwargs)
    fw_function_replacements = glob.wrapped_fns

    fn = to_reload
    is_callable = False
    ret = None
    if (
        (hasattr(fn, "__call__"))
        and not isinstance(fn, FunctionType)
        and not hasattr(fn, "__self__")
    ):
        is_callable = True

    def search_for_name(ns):
        # locate the function, callable or method within the reloaded namespace.
        if is_callable:
            reloaded_class = ns.callables[to_reload.__class__.__name__]
            object.__setattr__(to_reload, "__class__", reloaded_class)
            return to_reload
        elif hasattr(to_reload, "__self__"):
            # Because of class replacements, the instance
            # should have the new bound method.
            return safe_getattr(to_reload.__self__, to_reload.__name__)
        elif callable(to_reload):
            return ns.functions[to_reload.__name__]

    def reload_module(module):
        if module is not None:
            recursive_reload(module)

    def reload_modules(modules):
        for module in modules:
            reload_module(module)

    def get_ret(module):
        nonlocal ret

        try:
            ret = search_for_name(Namespace(module.__dict__))
        except KeyError:
            ret = to_reload
        if ret is None:
            ret = to_reload

    try:
        modules = get_modules(to_reload)
        # todo: raise something for non-imported classes
        reload_modules(modules)
        patch_higher_modules()
        # replace classes of existing instances
        replace_classes()
        # search args and kwargs
        args, kwargs = replace_args_kwargs(args, kwargs)
        # get ret
        ret_module = try_getmodule(to_reload)
        if ret_module is not None:
            ret = get_ret(ret_module)
        if ret is None:
            ret = to_reload
        # search ret itself
        ret = search_var(ret)
        # reload frameworks
        reload_frameworks()
        return ret
    except Exception as e:
        print("Reloading failed")
        raise e


def search_var(obj):
    """
    Recursively replace framework functions contained in
    mutable variables. For immutables, we return a copy
    which has been reloaded.
    """
    global searched_mutables
    ret = obj
    if isinstance(ret, (int, str)):
        return ret
    if ret is None:
        return None
    while id(ret) in fw_function_replacements:
        if ret is fw_function_replacements[id(ret)][1]:
            break
        ret = fw_function_replacements[id(ret)][1]
    if isinstance(obj, functools.partial):
        nfn = search_var(obj.func)
        nargs = search_var(obj.args)
        nkwargs = search_var(obj.keywords)
        ret = functools.partial(nfn, *nargs, **nkwargs)
        return ret
    if ret is not obj:
        return ret
    if id(obj) in searched_mutables:
        return obj
    searched_mutables[id(obj)] = True
    # Avoid replacing the __wrapped__ attribute of gc's logger fns
    if "wrapped_for_tracing" in ibdir(ret):
        return ret
    if isinstance(obj, type):
        return obj
    if isinstance(obj, list):
        for idx in range(len(obj)):
            item = obj[idx]
            new_item = search_var(item)
            if item is not new_item:
                obj[idx] = new_item
        return obj
    if isinstance(obj, dict):
        for key in dict.keys(obj):
            item = obj[key]
            new_item = search_var(item)
            if item is not new_item:
                obj[key] = new_item
        return obj
    if isinstance(obj, (FunctionType, MethodType)):
        if obj.__closure__ is not None:
            for cell in obj.__closure__:
                oval = cell.cell_contents
                nval = search_var(oval)
                if nval is not oval:
                    cell.cell_contents = nval
            return obj
    return obj


def replace_args_kwargs(args, kwargs):
    nargs = []
    nkwargs = {}
    for arg in args:
        nargs.append(search_var(arg))
    for k, v in kwargs.items():
        nkwargs[k] = search_var(v)
    return nargs, nkwargs


def replace_classes():
    """
    Apply search_var to all existing relevant objects.
    """
    global class_replacement

    objs = get_all_objects()

    for obj in objs:
        try:
            if not hasattr(obj, "__class__"):
                continue
        except ReferenceError:
            continue

        cls = obj.__class__

        if id(cls) in class_replacements or fast_getmodule(cls).startswith("keras"):
            if id(obj.__class__) in classes_to_ignore:
                continue

            for attr_name in ibdir(obj):
                attr = safe_getattr(obj, attr_name)
                try:
                    new_attr = search_var(attr)
                    if attr is not new_attr:
                        object.__setattr__(obj, attr_name, new_attr)
                    attr = safe_getattr(obj, attr_name)
                except Exception:
                    break

            if id(obj.__class__) not in class_replacements:
                continue


def get_all_objects():
    """
    Return a list of all live Python
    objects, not including the list itself.
    """
    gcl = gc.get_objects()
    olist = [*gcl]
    seen = {}
    # Just in case:
    seen[id(gcl)] = None
    seen[id(olist)] = None
    seen[id(seen)] = None

    slist = gcl
    while len(slist) > 0 and len(olist) + len(gcl) < MAX_OBJS:
        tl = []
        for e in slist:
            if id(e) in seen:
                continue
            seen[id(e)] = None
            olist.append(e)
            for r in gc.get_referents(e):
                tl.append(r)
        slist = tl

    return olist


def reload_frameworks():
    """
    Sometimes our initial wrapping code doesn't succeed in wrapping framework functions.
    We can use the reloader to clean up these cases.
    """
    global reloaded
    reloaded = {}

    def recurse(module):
        global reloaded
        if first_name(get_name(module)) not in modules_to_update:
            return
        if id(module) in reloaded:
            return
        for attr_name in ibdir(module):
            reloaded[id(module)] = True
            try:
                attr = getattr(module, attr_name)
            except:
                continue
            if isinstance(attr, ModuleType):
                recurse(attr)
                continue
            try:
                if isinstance(attr, type):
                    setattr(module, attr_name, search_var(attr))
                if isinstance(attr, FunctionType):
                    setattr(module, attr_name, search_var(attr))
            except:
                continue

    for module in visible_frameworks.values():
        recurse(module)


def recursive_reload(module):
    """
    Search module dependency tree, and reload modules which are dependent on
    a wrapped framework.
    """
    global reloaded, is_relevant
    ret = False
    if module is None:
        return False
    if first_name(get_name(module)) in modules_to_update:
        if id(module) not in visible_frameworks:
            visible_frameworks[id(module)] = module
        return True
    for attr_name in ibdir(module):
        try:
            attr = getattr(module, attr_name)
        except:
            continue

        if type(attr) is ModuleType:
            submodule = attr
        else:
            if not (callable(attr) or isinstance(attr, type)):
                continue
            submodule = inspect.getmodule(attr)

        if submodule is None:
            continue
        if (
            id(submodule) in reloaded
            and first_name(get_name(submodule)) not in modules_to_update
        ):
            continue
        if get_name(submodule) == get_name(module):
            continue
        if first_name(get_name(submodule)) in modules_to_ignore:
            continue
        if get_name(submodule) in is_relevant:
            ret = True
            continue

        reloaded[id(submodule)] = True

        if recursive_reload(submodule):
            ret = True

    if first_name(get_name(module)) in modules_to_ignore:
        return ret
    if ret:
        is_relevant[get_name(module)] = True
        try:
            reload(module)
        except Exception:
            print("failed to reload", module)
            print(traceback.format_exc())
    return ret


def is_classdef(module, obj):
    """
    Check that the ``obj`` is a class
    originating in ``module``.
    """
    if not isinstance(obj, type):
        return False
    if not hasattr(obj, "__module__"):
        return False
    if not obj.__module__ == get_name(module):
        return False
    return True


def is_function(module, obj):
    """
    Returns True iff ``obj`` is a function defined in ``module``.
    """
    if not isinstance(obj, (FunctionType, BuiltinFunctionType)):
        return False
    if not hasattr(obj, "__module__"):
        return False
    if not obj.__module__ == get_name(module):
        return False
    return True


def safe_getattr(obj, name):
    """
    Since getattr() sometimes doesn't work,
    for reasons out of our control.
    """
    if isinstance(obj, type):
        try:
            ret = type.__getattribute__(obj, name)
        except Exception:
            ret = None
        return ret
    try:
        ret = object.__getattribute__(obj, name)
    except Exception:
        ret = None
    return ret


def ibdir(obj):
    """
    More consistent than calling dir(), which can be
    overriden.
    """
    ret = []
    if isinstance(obj, type):
        try:
            return type.__dir__(obj)
        except:
            return dir(obj)
    try:
        ibd = object.__dir__(obj)
    except:
        return dir(obj)
    for k in ibd:
        if k != "__dict__" and k != "__dir__":
            ret.append(k)
    return ret


def try_getmodule(obj):
    try:
        module = inspect.getmodule(obj)
        return module
    except:
        return None


def fast_getmodule(obj):
    if not hasattr(obj, "__module__"):
        return ""
    return str(obj.__module__)


def exdir(obj):
    return [safe_getattr(obj, k) for k in ibdir(obj)]


def remove_none(list):
    return [x for x in list if x is not None]


def children(obj):
    if isinstance(obj, dict):
        return [*obj.values()]
    if isinstance(obj, list):
        return [*obj]
    if isinstance(obj, tuple):
        return [*obj]
    return ibdir(obj)


def get_modules(obj):
    """
    Get a list of modules referenced by ``obj``. As sometimes
    inspect.getmodule() returns the wrong module, we should
    reload all that are relevant.
    """
    v = {}
    mlist = [try_getmodule(obj)]
    n = exdir(obj)
    # How deep to search.
    generations = 2
    max_modules = 10
    for _ in range(generations):
        nn = []
        for o in n:
            if len(mlist) >= max_modules:
                break
            if id(o) in v:
                continue
            v[id(o)] = True
            if isinstance(o, ModuleType):
                mlist.append(o)
                continue
            m = try_getmodule(o)
            if m and id(m) not in v:
                mlist.append(m)
                v[id(m)] = True
            for c in exdir(o):
                if id(c) in v:
                    continue
                nn.append(c)
        n = nn
        if len(mlist) >= max_modules:
            break
    return mlist


def wrap_init(cls, name, oval, nval):
    """
    Update class variables. Some classes also need to have their
    __post_init__ patched (dataclasses require this).
    """
    type.__setattr__(cls, name, nval)
    try:
        original_init = type.__getattribute__(cls, "__post_init__")
    except AttributeError:
        return

    def __post_init__(self, *args, **kwargs):
        if isinstance(original_init, MethodType):
            original_init(*args, **kwargs)
        else:
            original_init(self, *args, **kwargs)
        if safe_getattr(self, name) is oval:
            object.__setattr__(self, name, nval)

    if isinstance(original_init, MethodType):
        __post_init__ = MethodType(__post_init__, cls)

    type.__setattr__(cls, "__post_init__", __post_init__)


def reload(module):
    """
    Applies function replacement to variables inside a module,
    modifying in-place.
    """
    global class_replacements, function_replacements
    classes = [
        safe_getattr(module, k)
        for k in ibdir(module)
        if is_classdef(module, safe_getattr(module, k))
    ]
    functions = [
        safe_getattr(module, k)
        for k in ibdir(module)
        if is_function(module, safe_getattr(module, k))
    ]
    global_instances = {}
    for k in ibdir(module):
        global_instances[k] = safe_getattr(module, k)

    for k in global_instances.keys():
        object.__setattr__(module, k, search_var(global_instances[k]))

    # This section hsa been known to have some strange behaviours
    # regarding replacing things with completely different things,
    # not sure what the root issue is.
    for c in classes:
        class_replacements[id(c)] = c
        for k in type.__dir__(c):
            if k == "__dict__":
                continue
            if k == "__post_init__":
                continue
            if k == "__init_subclass__":
                continue
            if k == "__subclasshook__":
                continue
            attr = safe_getattr(c, k)
            if in_fw_fn_replacements(attr):
                try:
                    wrap_init(c, k, attr, fw_function_replacements[id(attr)][1])
                except:
                    break

    for f in functions:
        function_replacements[id(f)] = f

    for attr_name in ibdir(module):
        attr = safe_getattr(module, attr_name)
        valid_callable = isinstance(
            attr, (FunctionType, BuiltinFunctionType, WrappedCallable)
        ) or is_framework_callable(attr)
        if valid_callable:
            try:
                object.__setattr__(module, attr_name, search_var(attr))
            except:
                break

    touched_module_ids[id(module)] = True

    return module


def is_framework_callable(obj):
    if id(obj) in functions_to_ignore:
        return False
    if not callable(obj):
        return False
    if hasattr(obj, "__class__"):
        if hasattr(obj.__class__, "__module__"):
            if first_name(obj.__class__.__module__) in modules_to_update:
                return True
    return False


def in_fw_fn_replacements(obj):
    if id(obj) in functions_to_ignore:
        return False
    if id(obj) in fw_function_replacements:
        return True
    return False


def alias_to_name(alias):
    if alias.asname is None:
        name = alias.name
    else:
        name = alias.asname
    name = trim_path(name)
    return ast.Name(id=name, ctx=ast.Load())


def trim_path(name):
    """
    Same as first_name but preserves the relative component.
    """
    leading = ""
    for c in name:
        if c != ".":
            break
        leading += c
    first = list(filter(lambda n: n != "", name.split(".")))[0]
    return leading + first


def find_module(name):
    """
    Lookup a module by its name.
    """
    if name in sys.modules:
        return sys.modules[name]
    return None


def first_name(name):
    """
    Returns the first name in a period-separated path.
    """
    return name.split(".")[0]


def join_path(lst):
    return ".".join(lst)


def get_name(obj):
    try:
        return obj.__name__
    except:
        return ""