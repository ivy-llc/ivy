# global
import os
import copy
import types
import ivy
import importlib
import functools
import numpy as np
import gc
from ivy.utils import _importlib

# local
from ivy.func_wrapper import _wrap_function
from ivy.utils.backend.sub_backend_handler import (
    _clear_current_sub_backends,
    fn_name_from_version_specific_fn_name,
)
from ivy.utils.exceptions import _handle_inplace_mode

backend_stack = []
compiled_backends = {}
_compiled_backends_ids = {}
implicit_backend = "numpy"
ivy_original_dict = ivy.__dict__.copy()
ivy_original_fn_dict = {}


class ContextManager:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        return set_backend(self.module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        previous_backend()


_backends_subpackage_path = "ivy.functional.backends"
_backend_dict = {}
_backend_reverse_dict = {}

for backend in os.listdir(
    os.path.join(
        ivy.__path__[0].rpartition(os.path.sep)[0],  # type: ignore
        _backends_subpackage_path.replace(".", os.path.sep),
    )
):
    if backend.startswith("__"):
        continue
    backend_path = f"{_backends_subpackage_path}.{backend}"
    _backend_dict[backend] = backend_path
    _backend_reverse_dict[backend_path] = backend


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
    for backend, module_name in _backend_dict.items():
        if backend in arg_module_name:
            return importlib.import_module(module_name)


def _determine_backend_from_args(args):
    """Return the appropriate Ivy backend, given some arguments.

    Parameters
    ----------
    args
        the arguments from which to figure out the corresponding Ivy backend.

    Returns
    -------
    ret
        the Ivy backend inferred from `args`.

    Examples
    --------
    If `args` is a jax.numpy array, then Ivy's jax backend will be returned:

    >>> from ivy.utils.backend.handler import _determine_backend_from_args
    >>> import jax.numpy as jnp
    >>> x = jnp.array([1])
    >>> print(_determine_backend_from_args(x))
    <module 'ivy.functional.backends.jax' from '/ivy/ivy/functional/backends/jax/__init__.py'>    # noqa
    """  # noqa: E501
    arg_type = type(args)
    if isinstance(args, ivy.Array):
        args = args.data

    if isinstance(args, dict):
        for value in args.values():
            # recursively call the function for each value in the dictionary
            lib = _determine_backend_from_args(value)
            if lib:
                return lib
        # check if args is a list or tuple
    elif arg_type in [list, tuple]:
        for arg in args:
            # recursively call the function for each element in the list/tuple
            lib = _determine_backend_from_args(arg)
            if lib:
                return lib
    else:
        # check if the class module of the arg is in _array_types
        return _get_backend_for_arg(args.__class__.__module__)


def set_backend_to_specific_version(backend):
    """Update the backend dict to make the original function name point to the
    version specific one.

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
    """Return the current backend. Priorities: global_backend > argument's
    backend.

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
    """  # noqa: E501
    global implicit_backend
    # if a global backend has been set with
    # set_backend then this will be returned
    if backend_stack:
        f = backend_stack[-1]
        return f

    # if no global backend exists, we try to infer
    # the backend from the arguments
    f = _determine_backend_from_args(list(args) + list(kwargs.values()))
    if f is not None:
        implicit_backend = f.current_backend_str()
        return f
    return importlib.import_module(_backend_dict[implicit_backend])


def _set_module_backend(
    original_dict, target, backend, invalid_dtypes=None, backend_str=None
):
    invalid_dtypes = (
        backend.invalid_dtypes if invalid_dtypes is None else invalid_dtypes
    )
    backend_str = backend.current_backend_str() if backend_str is None else backend_str
    for k, v in original_dict.items():
        if k in ivy.GLOBAL_PROPS:
            continue
        compositional = k not in backend.__dict__
        if compositional:
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
            _set_module_backend(
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


def _data_to_new_backend(x, previous_backend):
    device = previous_backend.dev(x.data)
    try:
        result = ivy.from_dlpack(previous_backend.to_dlpack(x.data))
        result = ivy.to_device(result, device)
    except Exception:
        np_res = previous_backend.to_numpy(x.data)
        result = ivy.asarray(np_res, device=device)
    return result


def dynamic_backend_converter(backend_stack):
    from ivy.functional.ivy.gradients import _variable

    def _is_var(obj, backend):
        if isinstance(obj, ivy.Container):

            def _map_fn(x):
                x = x.data if isinstance(x, ivy.Array) else x
                if x.__class__.__module__ in (
                    "numpy",
                    "jax.interpreters.xla",
                    "jaxlib.xla_extension",
                ):
                    return False

                return backend.gradients._is_variable(x)

            return obj.cont_map(lambda x, kc: _map_fn(x)).cont_all_true()

        else:
            obj = obj.data if isinstance(obj, ivy.Array) else obj
            if obj.__class__.__module__ in (
                "numpy",
                "jax.interpreters.xla",
                "jaxlib.xla_extension",
            ):
                return False
            return backend.gradients._is_variable(obj)

    # get all ivy array instances in the project scope
    container_list = [
        obj
        for obj in gc.get_objects()
        if "ivy" in type(obj).__module__ and isinstance(obj, ivy.Container)
    ]
    cont_array_idxs = ivy.nested_argwhere(
        container_list,
        lambda x: isinstance(x, ivy.Array) and x.backend != ivy.current_backend_str(),
    )
    cont_array_vals = ivy.multi_index_nest(container_list, cont_array_idxs)
    array_list = [
        obj
        for obj in gc.get_objects()
        if "ivy" in type(obj).__module__ and isinstance(obj, ivy.Array)
    ]
    array_list.extend(cont_array_vals)

    # filter uninitialized arrays and arrays with other backends, and ensure the order
    array_list = [
        arr
        for arr in array_list
        if arr.__dict__ and arr.backend != ivy.current_backend_str()
    ]
    new_objs = [obj for obj in array_list if obj.dynamic_backend]

    # now convert all ivy.Array and ivy.Container instances
    # to the new backend

    for obj in new_objs:
        # the following if condition avoids converting arrays that were already
        # updated inplace i.e. are references to other arrays
        if obj.backend != ivy.current_backend_str():
            backend = ivy.with_backend(obj.backend, cached=True)
            if _is_var(obj, backend):
                native_var = backend.gradients._variable_data(obj)
                data = _data_to_new_backend(native_var, backend)
                new_data = _variable(data)

            else:
                new_data = _data_to_new_backend(obj, backend)

            if isinstance(obj, ivy.Container):
                obj.cont_inplace_update(new_data)
            else:
                obj.data = new_data.data


@prevent_access_locally
def set_backend(backend: str, dynamic: bool = False):
    """Set `backend` to be the global backend.

    Will also convert all Array and Container objects to the new backend if `dynamic` =
    True

    Examples
    --------
    If we set the global backend to be numpy, then subsequent calls to ivy functions
    will be called from Ivy's numpy backend:

    >>> ivy.set_backend("numpy")
    >>> native = ivy.native_array([1])
    >>> print(type(native))
    <class 'numpy.ndarray'>

    Or with jax as the global backend:

    >>> ivy.set_backend("jax")
    >>> native = ivy.native_array([1])
    >>> print(type(native))
    <class 'jaxlib.xla_extension.ArrayImpl'>
    """  # noqa
    ivy.utils.assertions.check_false(
        isinstance(backend, str) and backend not in _backend_dict,
        f"backend must be one from {list(_backend_dict.keys())}",
    )

    # update the global dict with the new backend
    with ivy.locks["backend_setter"]:
        global ivy_original_dict
        if not backend_stack:
            ivy_original_dict = ivy.__dict__.copy()
        _clear_current_sub_backends()
        if isinstance(backend, str):
            temp_stack = []
            while backend_stack:
                temp_stack.append(previous_backend())
            backend = importlib.import_module(_backend_dict[backend])
            for fw in reversed(temp_stack):
                backend_stack.append(fw)
        if backend.current_backend_str() == "numpy":
            ivy.set_default_device("cpu")
        elif backend.current_backend_str() == "jax":
            ivy.set_global_attr("RNG", ivy.functional.backends.jax.random.RNG)
        backend_stack.append(backend)
        set_backend_to_specific_version(backend)
        _set_module_backend(ivy_original_dict, ivy, backend)
        # following snippet is required to update the ivy.functional namespace with
        # backend-specific functions
        for key in ivy.__dict__.keys():
            if key in ivy.functional.__dict__ and not key.startswith("__"):
                ivy.functional.__dict__[key] = ivy.__dict__[key]

        if dynamic:
            dynamic_backend_converter(backend_stack)
        for sub_backend in ivy.available_sub_backends:
            ivy.set_sub_backend(sub_backend)
    _handle_inplace_mode()
    return ivy


def set_numpy_backend():
    """Set NumPy to be the global backend.

    equivalent to `ivy.set_backend("numpy")`.
    """  # noqa
    set_backend("numpy")


def set_jax_backend():
    """Set JAX to be the global backend.

    equivalent to `ivy.set_backend("jax")`.
    """  # noqa
    set_backend("jax")


def set_tensorflow_backend():
    """Set TensorFlow to be the global backend.

    equivalent to `ivy.set_backend("tensorflow")`.
    """
    set_backend("tensorflow")


def set_torch_backend():
    """Set torch to be the global backend.

    equivalent to `ivy.set_backend("torch")`.
    """  # noqa
    set_backend("torch")


def set_paddle_backend():
    """Set paddle to be the global backend.

    equivalent to `ivy.set_backend("paddle")`.
    """  # noqa
    set_backend("paddle")


def set_mxnet_backend():
    """Set MXNet to be the global backend.

    equivalent to `ivy.set_backend("mx")`.
    """  # noqa
    set_backend("mxnet")


@prevent_access_locally
def previous_backend():
    """Unset the current global backend, and adjusts the ivy dict such that
    either a previously set global backend is then used as the backend,
    otherwise we return to Ivy's implementations.

    Returns
    -------
    ret
        the backend that was unset, or None if there was no set global backend.

    Examples
    --------
    Torch is the last set backend hence is the backend used in the first examples.
    However, as seen in the example after, if `previous_backend` is called before
    `ivy.native_array` then tensorflow will become the current backend and any
    torch backend implementations in the Ivy dict will be swapped with the
    tensorflow implementation::

    >>> ivy.set_backend("tensorflow")
    >>> ivy.set_backend("torch")
    >>> x = ivy.native_array([1])
    >>> print(type(x))
    <class 'torch.Tensor'>

    >>> ivy.set_backend("tensorflow")
    >>> ivy.set_backend("torch")
    >>> ivy.previous_backend()
    >>> x = ivy.native_array([1])
    >>> print(type(x))
    <class'tensorflow.python.framework.ops.EagerTensor'>
    """  # noqa
    backend = None
    # if the backend stack is empty, nothing is done then we just return `None`
    if backend_stack:
        backend = backend_stack.pop(-1)  # remove last backend from the stack
        if backend.current_backend_str() == "numpy":
            ivy.unset_default_device()
        elif backend.current_backend_str() == "jax":
            ivy.del_global_attr("RNG")
        # the new backend is the backend that was set before the one
        # we just removed from the stack, or Ivy if there was no
        # previously set backend
        if backend_stack:
            new_backend = backend_stack[-1]
            if new_backend.current_backend_str() == "numpy":
                ivy.set_default_device("cpu")
            elif new_backend.current_backend_str() == "jax":
                ivy.set_global_attr("RNG", ivy.functional.backends.jax.random.RNG)
        new_backend_dict = (
            backend_stack[-1].__dict__ if backend_stack else ivy_original_dict
        )
        # wrap backend functions if there still is a backend, and add functions
        # to ivy namespace
        for k, v in new_backend_dict.items():
            if k in ivy.GLOBAL_PROPS:
                continue
            if backend_stack and k in ivy_original_dict:
                v = _wrap_function(k, v, ivy_original_dict[k])
            if k in ivy_original_dict:
                ivy.__dict__[k] = v
            if k in ivy.functional.__dict__ and not k.startswith("__"):
                ivy.functional.__dict__[k] = v
    _handle_inplace_mode()
    return backend


@prevent_access_locally
def unset_backend():
    while backend_stack:
        previous_backend()


@prevent_access_locally
def choose_random_backend(excluded=None):
    excluded = [] if excluded is None else excluded
    while True:
        ivy.utils.assertions.check_equal(
            len(excluded),
            4,
            inverse=True,
            message="""Unable to select backend, all backends are excluded,\
            or not installed.""",
            as_array=False,
        )
        f = np.random.choice(
            [f_srt for f_srt in list(_backend_dict.keys()) if f_srt not in excluded]
        )
        if f is None:
            excluded.append(f)
            continue
        else:
            print(f"\nselected backend: {f}\n")
            return f


# noinspection PyProtectedMember
@prevent_access_locally
def with_backend(backend: str, cached: bool = True):
    # Use already compiled object
    if cached and backend in compiled_backends:
        cached_backend = compiled_backends[backend][-1]
        return cached_backend
    with _importlib.LocalIvyImporter():
        ivy_pack = _importlib._import_module("ivy")
        ivy_pack._is_local_pkg = True
        ivy_pack._compiled_id = id(ivy_pack)
        backend_module = _importlib._import_module(
            ivy_pack.utils.backend.handler._backend_dict[backend], ivy_pack.__package__
        )
        _handle_backend_specific_vars(ivy_pack, backend_module)
        set_backend_to_specific_version(backend_module)
        # We know for sure that the backend stack is empty
        # no need to do backend unsetting
        ivy_pack.utils.backend.handler._set_module_backend(
            ivy_pack.__dict__.copy(), ivy_pack, backend_module
        )
        # TODO use a refactored code from ivy.set_backend
        for key in ivy_pack.__dict__.keys():
            if key in ivy_pack.functional.__dict__ and not key.startswith("__"):
                ivy_pack.functional.__dict__[key] = ivy_pack.ivy.__dict__[key]
        ivy_pack.backend_stack.append(backend_module)
        ivy_pack.utils.backend._importlib.import_cache = copy.copy(
            _importlib.import_cache
        )
        _compiled_backends_ids[ivy_pack._compiled_id] = ivy_pack
        _importlib._clear_cache()
    try:
        compiled_backends[backend].append(ivy_pack)
    except KeyError:
        compiled_backends[backend] = [ivy_pack]
    if ivy.backend != backend:
        # to avoid warning users when not using set_backend with ivy.Array.__repr__
        _handle_inplace_mode(ivy_pack=ivy_pack)
    return ivy_pack
