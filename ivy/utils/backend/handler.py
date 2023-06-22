# global
import os
import copy
import types
import ivy
import importlib
import functools
import numpy as np
import gc
from ivy.utils import _importlib, verbosity

# local
from ivy.func_wrapper import _wrap_function
from ivy.utils.backend.sub_backend_handler import _clear_current_sub_backends

backend_stack = []
compiled_backends = {}
_compiled_backends_ids = {}
implicit_backend = "numpy"
ivy_original_dict = ivy.__dict__.copy()
ivy_original_fn_dict = dict()

class ContextManager:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        return set_backend(self.module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        previous_backend()

_backends_subpackage_path = "ivy.functional.backends"
_backend_dict = dict()
_backend_reverse_dict = dict()

for backend in os.listdir(os.path.join(ivy.__path__[0].rpartition(os.path.sep)[0], _backends_subpackage_path.replace(".", os.path.sep))):
    if backend.startswith("__"):
        continue
    backend_path = f"{_backends_subpackage_path}.{backend}"
    _backend_dict[backend] = backend_path
    _backend_reverse_dict[backend_path] = backend

def set_backend_to_specific_version(backend):
    """
    Update the backend dict to make the original function name point to the version
    specific one.
    Parameters
    ----------
    backend
        the backend module for which we provide the version support
    """
    # TODO: add functionality and tests
    f = str(backend.__name__)
    f = f[f.index("backends") + 9 :]

    f = importlib.import_module(f)
    f_version = f.__version__

    for key in list(backend.__dict__):
        if "_v_" in key:
            orig_name = fn_name_from_version_specific_fn_name(key, f_version)
            if orig_name:
                backend.__dict__[orig_name] = backend.__dict__[key]
                backend.__dict__[orig_name].__name__ = orig_name


def current_backend(*args, **kwargs):
    """
    Return the current backend. Priorities: global_backend > argument's backend.
    Parameters
    ----------
    *args/**kwargs
        the arguments from which to try to infer the backend, when there is
        no globally set backend.
    Returns
    -------
    ret
        Ivy's current backend.
    Examples
    --------
    If no global backend is set, then the backend is inferred from the arguments:
    >>> import numpy as np
    >>> x = np.array([2.0])
    >>> print(ivy.current_backend(x))
    <module 'ivy.functional.backends.numpy' from '/ivy/ivy/functional/backends/numpy/__init__.py'>   # noqa
    The global backend set in set_backend has priority over any arguments
    passed to current_backend:
    >>> import numpy as np
    >>> ivy.set_backend("jax")
    >>> x = np.array([2.0])
    >>> print(ivy.current_backend(x))
    <module 'ivy.functional.backends.jax' from '/ivy/ivy/functional/backends/jax/__init__.py'>   # noqa
    """
    global implicit_backend
    # if a global backend has been set with
    # set_backend then this will be returned
    if backend_stack:
        f = backend_stack[-1]
        if verbosity.level > 0:
            verbosity.cprint("Using backend from stack: {}".format(f))
        return f

    # if no global backend exists, we try to infer
    # the backend from the arguments
    f = _determine_backend_from_args(list(args) + list(kwargs.values()))
    if f is not None:
        implicit_backend = f.current_backend_str()
        return f
    if verbosity.level > 0:
        verbosity.cprint("Using backend from type: {}".format(f))
    return importlib.import_module(_backend_dict[implicit_backend])


def _set_backend_as_ivy(
    original_dict, target, backend, invalid_dtypes=None, backend_str=None
):
    invalid_dtypes = (
        backend.invalid_dtypes if invalid_dtypes is None else invalid_dtypes
    )
    backend_str = backend.current_backend_str() if backend_str is None else backend_str
    for k, v in original_dict.items():
        compositional = k not in backend.__dict__
        if k not in backend.__dict__:
            if k in invalid_dtypes and k in target.__dict__:
                del target.__dict__[k]
                continue
            backend.__dict__[k] = v
        target.__dict__[k] = _wrap_function(
            key=k, to_wrap=backend.__dict__[k], original=v, compositional=compositional
        )
        if (
            isinstance(v, types.ModuleType)
            and "ivy.functional." in v.__name__
            and os.path.join("{}", "__init__.py").format(backend_str) not in v.__file__
        ):
            _set_backend_as_ivy(
                v.__dict__,
                target.__dict__[k],
                backend.__dict__[k],
                invalid_dtypes=invalid_dtypes,
                backend_str=backend_str,
            )


def _handle_backend_specific_vars(target, backend):
    if backend.current_backend_str() == "numpy":
        target.set_default_device("cpu")
    elif backend.current_backend_str() == "jax":
        target.set_global_attr("RNG", target.functional.backends.jax.random.RNG)


def convert_from_source_backend_to_numpy(variable_ids, numpy_objs, devices):
    # Dynamic Backend
    from ivy.functional.ivy.gradients import _is_variable, _variable_data

    def _is_var(obj):
        if isinstance(obj, ivy.Container):

            def _map_fn(x):
                x = x.data if isinstance(x, ivy.Array) else x
                if x.__class__.__module__ in (
                    "numpy",
                    "jax.interpreters.xla",
                    "jaxlib.xla_extension",
                ):
                    return False

                return _is_variable(x)

            return obj.cont_map(lambda x, kc: _map_fn(x)).cont_all_true()

# Backend Getting/Setting #
# ----------------------- #


def prevent_access_locally(fn):
    @functools.wraps(fn)
    def _prevent_access_locally(*args, **kwargs):
        if ivy.is_local():
            raise RuntimeError(f"Calling {fn.__name__} is not allowed on this object.")
        return fn(*args, **kwargs)

    return _prevent_access_locally


@functools.lru_cache
def _get_backend_for_arg(arg_module_name):
    for backend in _backend_dict:
        if backend in arg_module_name:
            module_name = _backend_dict[backend]
            return importlib.import_module(module_name)


def _determine_backend_from_args(args):
    arg_type = type(args)
    if isinstance(args, ivy.Array):
        args = args.data

    if isinstance(args, dict):
        for key, value in args.items():
            lib = _determine_backend_from_args(value)
            if lib:
                return lib
    elif arg_type in [list, tuple]:
        for arg in args:
            lib = _determine_backend_from_args(arg)
            if lib:
                return lib
    else:
        if not hasattr(args, "ivy_array"):
            return _get_backend_for_arg(args.__class__.__module__)


def fn_name_from_version_specific_fn_name(name, version):
    version = str(version)
    if version.find("+") != -1:
        version = tuple(map(int, version[: version.index("+")].split(".")))
    else:
        version = tuple(map(int, version.split(".")))
    if "_to_" in name:
        i = name.index("_v_")
        e = name.index("_to_")
        version_start = name[i + 3 : e]
        version_start = tuple(map(int, version_start.split("p")))
        version_end = name[e + 4 :]
        version_end = tuple(map(int, version_end.split("p")))
        if version_start <= version <= version_end:
            return name[0:i]
    elif "_and_above" in name:
        i = name.index("_v_")
        e = name.index("_and_")
        version_start = name[i + 3 : e]
        version_start = tuple(map(int, version_start.split("p")))
        version_end = name[e + 5 :]
        version_end = tuple(map(int, version_end.split("p")))
        if version_start <= version:
            return name[0:i]
    else:
        i = name.index("_v_")
        version_start = name[i + 3 :]
        version_start = tuple(map(int, version_start.split("p")))
        if version_start == version:
            return name[0:i]
    return None


@prevent_access_locally
def get_backend(backend_str=None):
    if backend_str is None:
        if backend_stack:
            return backend_stack[-1]
        else:
            return implicit_backend

    if backend_str == "current":
        return get_current_sub_backend()

    backend_str = str(backend_str)

    if backend_str in _backend_dict:
        backend_str = _backend_dict[backend_str]

    if backend_str in _backend_reverse_dict:
        backend_str = _backend_reverse_dict[backend_str]

    if backend_str in compiled_backends:
        return backend_str

    raise Exception("Unsupported backend '{}' requested.".format(backend_str))


@prevent_access_locally
def set_backend(backend_str=None, version_str=None):
    global implicit_backend
    backend = get_backend(backend_str)

    if version_str is not None:
        backend = "{}_v_{}".format(backend, version_str)

    if not _backend_check(backend):
        raise Exception("The backend '{}' could not be set as the current backend.".format(backend))

    backend_stack.append(backend)
    return backend


@prevent_access_locally
def set_compile_backend_fn(backend_fn, backend_str=None):
    if backend_str is None:
        backend_str = get_backend()

    if backend_str not in compiled_backends:
        compiled_backends[backend_str] = types.SimpleNamespace()

    if isinstance(backend_fn, types.FunctionType):
        backend_fn = _wrap_function(backend_fn, exclude=["tvm"])

    backend_fn_module_name = backend_fn.__module__.split(".")[0]

    setattr(compiled_backends[backend_str], backend_fn_module_name, backend_fn)


@prevent_access_locally
def set_compile_backend_fn_versions(backend_fn_versions, backend_str=None):
    if backend_str is None:
        backend_str = get_backend()

    if backend_str not in compiled_backends:
        compiled_backends[backend_str] = types.SimpleNamespace()

    for backend_fn_version_key in backend_fn_versions:
        backend_fn_version = backend_fn_versions[backend_fn_version_key]
        set_compile_backend_fn(backend_fn_version, backend_str)


@prevent_access_locally
def get_current_sub_backend():
    if backend_stack:
        current_backend = backend_stack[-1]
        if current_backend in compiled_backends:
            return current_backend
    raise Exception("No valid backend has been set. "
                    "Either set a backend using ivy.set_backend or ivy.set_default_backend.")


@prevent_access_locally
def get_current_backend_fn(fn_name):
    backend = get_current_sub_backend()
    if hasattr(compiled_backends[backend], fn_name):
        return getattr(compiled_backends[backend], fn_name)
    return None


@prevent_access_locally
def get_backend_fn_versions(fn_name, backend_str=None):
    backend = get_backend(backend_str)

    if backend in compiled_backends:
        backend_fn = getattr(compiled_backends[backend], fn_name, None)
        if backend_fn:
            if hasattr(backend_fn, "versions"):
                return backend_fn.versions
            else:
                return dict()
    return dict()


@prevent_access_locally
def _backend_check(backend_str):
    backend_str = str(backend_str)
    if backend_str in compiled_backends:
        return True
    return False


@prevent_access_locally
def previous_backend():
    if backend_stack:
        return backend_stack.pop()
    return implicit_backend


@prevent_access_locally
def has_backend_fn(fn_name):
    backend = get_current_sub_backend()
    return hasattr(compiled_backends[backend], fn_name)


@prevent_access_locally
def clear_compiled_backends():
    compiled_backends.clear()
    _compiled_backends_ids.clear()
    ivy.__dict__.clear()
    ivy.__dict__.update(ivy_original_dict)
    gc.collect()


@prevent_access_locally
def clear_import_cache():
    _importlib.clear_cache()


@prevent_access_locally
def reload_backend_modules():
    _clear_current_sub_backends()
    for backend_str in compiled_backends:
        backend_module = _importlib.import_module(backend_str)
        ivy.__dict__.update({name: value for name, value in backend_module.__dict__.items()
                             if not name.startswith('_') and not isinstance(value, types.ModuleType)})
        for fn_name in dir(compiled_backends[backend_str]):
            if not fn_name.startswith("_"):
                if fn_name not in ivy.__dict__:
                    fn = getattr(compiled_backends[backend_str], fn_name)
                    if isinstance(fn, types.FunctionType):
                        ivy.__dict__[fn_name] = fn

    global ivy_original_fn_dict
    ivy_original_fn_dict.clear()
    ivy_original_fn_dict.update({name: value for name, value in ivy.__dict__.items()
                                 if not name.startswith('_') and isinstance(value, types.FunctionType)})
