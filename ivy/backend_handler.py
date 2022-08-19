# global
import ivy
import logging
import importlib
import numpy as np
from ivy import verbosity
from typing import Optional

# local
from ivy.func_wrapper import _wrap_function


backend_stack = []
implicit_backend = "numpy"
ivy_original_dict = ivy.__dict__.copy()
ivy_original_fn_dict = dict()


class ContextManager:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        set_backend(self.module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_backend()


_array_types = dict()
_array_types["numpy"] = "ivy.functional.backends.numpy"
_array_types["jax.interpreters.xla"] = "ivy.functional.backends.jax"
_array_types["jaxlib.xla_extension"] = "ivy.functional.backends.jax"
_array_types["tensorflow.python.framework.ops"] = "ivy.functional.backends.tensorflow"
_array_types["torch"] = "ivy.functional.backends.torch"
_array_types["mxnet.ndarray.ndarray"] = "ivy.functional.backends.mxnet"

_backend_dict = dict()
_backend_dict["numpy"] = "ivy.functional.backends.numpy"
_backend_dict["jax"] = "ivy.functional.backends.jax"
_backend_dict["tensorflow"] = "ivy.functional.backends.tensorflow"
_backend_dict["torch"] = "ivy.functional.backends.torch"
_backend_dict["mxnet"] = "ivy.functional.backends.mxnet"

_backend_reverse_dict = dict()
_backend_reverse_dict["ivy.functional.backends.numpy"] = "numpy"
_backend_reverse_dict["ivy.functional.backends.jax"] = "jax"
_backend_reverse_dict["ivy.functional.backends.tensorflow"] = "tensorflow"
_backend_reverse_dict["ivy.functional.backends.torch"] = "torch"
_backend_reverse_dict["ivy.functional.backends.mxnet"] = "mxnet"


# Backend Getting/Setting #
# ------------------------#


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

    >>> from ivy.backend_handler import _determine_backend_from_args
    >>> import jax.numpy as jnp
    >>> x = jnp.array([1])
    >>> print(_determine_backend_from_args(x))
    <module 'ivy.functional.backends.jax' from '/ivy/ivy/functional/backends/jax/__init__.py'>    # noqa

    """
    for arg in args:
        arg_type = type(arg)
        # function is called recursively if arg is a list/tuple
        if arg_type in [list, tuple]:
            lib = _determine_backend_from_args(arg)
            if lib:
                return lib
        # function is called recursively if arg is a dict
        elif arg_type is dict:
            lib = _determine_backend_from_args(list(arg.values()))
            if lib:
                return lib
        else:
            # use the _array_types dict to map the module where arg comes from, to the
            # corresponding Ivy backend
            if arg.__class__.__module__ in _array_types:
                module_name = _array_types[arg.__class__.__module__]
                return importlib.import_module(module_name)


def current_backend(*args, **kwargs):
    """Returns the current backend. Priorities:
    global_backend > argument's backend.

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
    # if a global backend has been set with set_backend then this will be returned
    if backend_stack:
        f = backend_stack[-1]
        if verbosity.level > 0:
            verbosity.cprint("Using backend from stack: {}".format(f))
        return f

    # if no global backend exists, we try to infer the backend from the arguments
    f = _determine_backend_from_args(list(args) + list(kwargs.values()))
    if f is not None:
        implicit_backend = f.current_backend_str()
        return f
    if verbosity.level > 0:
        verbosity.cprint("Using backend from type: {}".format(f))
    return importlib.import_module(_backend_dict[implicit_backend])


def set_backend(backend: str):
    """Sets `backend` to be the global backend.

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
    <class 'jaxlib.xla_extension.DeviceArray'>

    """
    if isinstance(backend, str) and backend not in _backend_dict:
        raise ValueError(
            "backend must be one from {}".format(list(_backend_dict.keys()))
        )
    ivy.locks["backend_setter"].acquire()
    global ivy_original_dict
    if not backend_stack:
        ivy_original_dict = ivy.__dict__.copy()
    if isinstance(backend, str):
        temp_stack = list()
        while backend_stack:
            temp_stack.append(unset_backend())
        backend = importlib.import_module(_backend_dict[backend])
        for fw in reversed(temp_stack):
            backend_stack.append(fw)
    if backend.current_backend_str() == "numpy":
        ivy.set_default_device("cpu")
    backend_stack.append(backend)

    for k, v in ivy_original_dict.items():
        if k not in backend.__dict__:
            if k in backend.invalid_dtypes and k in ivy.__dict__:
                del ivy.__dict__[k]
                continue
            backend.__dict__[k] = v
        ivy.__dict__[k] = _wrap_function(key=k, to_wrap=backend.__dict__[k], original=v)

    if verbosity.level > 0:
        verbosity.cprint("backend stack: {}".format(backend_stack))
    ivy.locks["backend_setter"].release()


def get_backend(backend: Optional[str] = None):
    """Returns Ivy's backend for `backend` if specified, or if it isn't specified it
    returns the Ivy backend associated with the current globally set backend.

    Parameters
    ----------
    backend
        The backend for which we want to retrieve Ivy's backend i.e. one of 'jax',
        'torch', 'tensorflow', 'numpy', 'mxnet'.

    Returns
    -------
    ret
        Ivy's backend for either `backend` or for the current global backend.

    Examples
    --------
    Global backend doesn't matter, if `backend` argument has been specified:

    >>> ivy.set_backend("jax")
    >>> ivy_np = ivy.get_backend("numpy")
    >>> print(ivy_np)
    <module 'ivy.functional.backends.numpy' from '/ivy/ivy/functional/backends/numpy/__init__.py'>   # noqa

    If backend isn't specified, the global backend is used:

    >>> ivy.set_backend("jax")
    >>> ivy_jax = ivy.get_backend()
    >>> print(ivy_jax)
    <module 'ivy.functional.backends.jax' from '/ivy/ivy/functional/backends/jax/__init__.py'>   # noqa

    """
    # ToDo: change this so that it doesn't depend at all on the global ivy. Currently
    #  all backend-agnostic implementations returned in this module will still
    #  use the global ivy backend.
    global ivy_original_dict
    if not backend_stack:
        ivy_original_dict = ivy.__dict__.copy()
    # current global backend is retrieved if backend isn't specified,
    # otherwise `backend` argument will be used
    if backend is None:
        backend = ivy.current_backend()
        if not backend_stack:
            return ""
    elif isinstance(backend, str):
        backend = importlib.import_module(_backend_dict[backend])
    for k, v in ivy_original_dict.items():
        if k not in backend.__dict__:
            backend.__dict__[k] = v
    return backend


def unset_backend():
    """Unsets the current global backend, and adjusts the ivy dict such that either
    a previously set global backend is then used as the backend, otherwise we return
    to Ivy's implementations.

    Returns
    -------
    ret
        the backend that was unset, or None if there was no set global backend.

    Examples
    --------
    Torch is the last set backend hence is the backend backend used here:

    >>> ivy.set_backend("tensorflow")
    >>> ivy.set_backend("torch")
    >>> x = ivy.native_array([1])
    >>> print(type(x))
    <class 'torch.Tensor'>

    However if `unset_backend` is called before `ivy.native_array` then tensorflow
    will become the current backend and any torch backend implementations in the
    Ivy dict will be swapped with the tensorflow implementation:

    >>> ivy.set_backend("tensorflow")
    >>> ivy.set_backend("torch")
    >>> ivy.unset_backend()
    >>> x = ivy.native_array([1])
    >>> print(type(x))
    <class'tensorflow.python.framework.ops.EagerTensor'>

    """
    backend = None
    # if the backend stack is empty, nothing is done and we just return `None`
    if backend_stack:
        backend = backend_stack.pop(-1)  # remove last backend from the stack
        if backend.current_backend_str() == "numpy":
            ivy.unset_default_device()
        # the new backend is the backend that was set before the one we just removed
        # from the stack, or Ivy if there was no previously set backend
        new_backend_dict = (
            backend_stack[-1].__dict__ if backend_stack else ivy_original_dict
        )
        # wrap backend functions if there still is a backend, and add functions
        # to ivy namespace
        for k, v in new_backend_dict.items():
            if backend_stack and k in ivy.__dict__:
                v = _wrap_function(k, v, ivy.__dict__[k])
            ivy.__dict__[k] = v
    if verbosity.level > 0:
        verbosity.cprint("backend stack: {}".format(backend_stack))
    return backend


def clear_backend_stack():
    while backend_stack:
        unset_backend()


# Backend Getters #
# ----------------#


def try_import_ivy_jax(warn=False):
    try:
        import ivy.functional.backends.jax

        return ivy.functional.backends.jax
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning(
            "{}\n\nEither jax or jaxlib appear to not be installed, "
            "ivy.functional.backends.jax can therefore not be imported.\n".format(e)
        )


def try_import_ivy_tf(warn=False):
    try:
        import ivy.functional.backends.tensorflow

        return ivy.functional.backends.tensorflow
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning(
            "{}\n\ntensorflow does not appear to be installed, "
            "ivy.functional.backends.tensorflow can therefore not be "
            "imported.\n".format(e)
        )


def try_import_ivy_torch(warn=False):
    try:
        import ivy.functional.backends.torch

        return ivy.functional.backends.torch
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning(
            "{}\n\ntorch does not appear to be installed, "
            "ivy.functional.backends.torch can therefore not be imported.\n".format(e)
        )


def try_import_ivy_mxnet(warn=False):
    try:
        import ivy.functional.backends.mxnet

        return ivy.functional.backends.mxnet
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning(
            "{}\n\nmxnet does not appear to be installed, "
            "ivy.functional.backends.mxnet can therefore not be imported.\n".format(e)
        )


def try_import_ivy_numpy(warn=False):
    try:
        import ivy.functional.backends.numpy

        return ivy.functional.backends.numpy
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning(
            "{}\n\nnumpy does not appear to be installed, "
            "ivy.functional.backends.numpy can therefore not be imported.\n".format(e)
        )


FW_DICT = {
    "jax": try_import_ivy_jax,
    "tensorflow": try_import_ivy_tf,
    "torch": try_import_ivy_torch,
    "mxnet": try_import_ivy_mxnet,
    "numpy": try_import_ivy_numpy,
}


def choose_random_backend(excluded=None):
    excluded = list() if excluded is None else excluded
    while True:
        if len(excluded) == 5:
            raise Exception(
                "Unable to select backend, all backends are either excluded "
                "or not installed."
            )
        f = np.random.choice(
            [f_srt for f_srt in list(FW_DICT.keys()) if f_srt not in excluded]
        )
        if f is None:
            excluded.append(f)
            continue
        else:
            print("\nselected backend: {}\n".format(f))
            return f
