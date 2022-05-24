# global
import ivy
import logging
import importlib
import collections
import numpy as np
from ivy import verbosity
from typing import Optional

# local
# noinspection PyProtectedMember
from ivy.func_wrapper import _wrap_functions, _unwrap_functions


framework_stack = []
ivy_original_dict = ivy.__dict__.copy()
ivy_original_fn_dict = dict()


class ContextManager:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        set_framework(self.module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_framework()


_array_types = dict()
_array_types["numpy"] = "ivy.functional.backends.numpy"
_array_types["jax.interpreters.xla"] = "ivy.functional.backends.jax"
_array_types["jaxlib.xla_extension"] = "ivy.functional.backends.jax"
_array_types["tensorflow.python.framework.ops"] = "ivy.functional.backends.tensorflow"
_array_types["torch"] = "ivy.functional.backends.torch"
_array_types["mxnet.ndarray.ndarray"] = "ivy.functional.backends.mxnet"

_framework_dict = dict()
_framework_dict["numpy"] = "ivy.functional.backends.numpy"
_framework_dict["jax"] = "ivy.functional.backends.jax"
_framework_dict["tensorflow"] = "ivy.functional.backends.tensorflow"
_framework_dict["torch"] = "ivy.functional.backends.torch"
_framework_dict["mxnet"] = "ivy.functional.backends.mxnet"

_framework_reverse_dict = dict()
_framework_reverse_dict["ivy.functional.backends.numpy"] = "numpy"
_framework_reverse_dict["ivy.functional.backends.jax"] = "jax"
_framework_reverse_dict["ivy.functional.backends.tensorflow"] = "tensorflow"
_framework_reverse_dict["ivy.functional.backends.torch"] = "torch"
_framework_reverse_dict["ivy.functional.backends.mxnet"] = "mxnet"


# Framework Getting/Setting #
# --------------------------#


def _determine_framework_from_args(args):
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

    >>> from ivy.framework_handler import _determine_framework_from_args
    >>> import jax.numpy as jnp
    >>> x = jnp.array([1])
    >>> print(_determine_framework_from_args(x))
    <module 'ivy.functional.backends.jax' from '/ivy/ivy/functional/backends/jax/__init__.py'>    # noqa

    """
    for arg in args:
        arg_type = type(arg)
        # function is called recursively if arg is a list/tuple
        if arg_type in [list, tuple]:
            lib = _determine_framework_from_args(arg)
            if lib:
                return lib
        # function is called recursively if arg is a dict
        elif arg_type is dict:
            lib = _determine_framework_from_args(list(arg.values()))
            if lib:
                return lib
        else:
            # use the _array_types dict to map the module where arg comes from, to the
            # corresponding Ivy backend
            if arg.__class__.__module__ in _array_types:
                module_name = _array_types[arg.__class__.__module__]
                return importlib.import_module(module_name)


def current_framework(*args, **kwargs):
    """Returns the current backend framework. Priorities:
    global_framework > argument's framework.

    Parameters
    ----------
    *args/**kwargs
        the arguments from which to try to infer the backend framework, when there is
        no globally set framework.

    Returns
    -------
    ret
        Ivy's current backend framework.

    Examples
    --------
    If no global framework is set, then the framework is inferred from the arguments:
    >>> import numpy as np
    >>> x = np.array([2.0])
    >>> print(ivy.current_framework(x))
    <module 'ivy.functional.backends.numpy' from '/ivy/ivy/functional/backends/numpy/__init__.py'>   # noqa

    The global framework set in set_framework has priority over any arguments
    passed to current_framework:
    >>> import numpy as np
    >>> ivy.set_framework("jax")
    >>> x = np.array([2.0])
    >>> print(ivy.current_framework(x))
    <module 'ivy.functional.backends.jax' from '/ivy/ivy/functional/backends/jax/__init__.py'>   # noqa

    """
    # if a global framework has been set with set_framework then this will be returned
    if framework_stack:
        f = framework_stack[-1]
        if verbosity.level > 0:
            verbosity.cprint("Using framework from stack: {}".format(f))
        return f

    # if no global framework exists, we try to infer the framework from the arguments
    f = _determine_framework_from_args(list(args) + list(kwargs.values()))
    if f is None:
        raise ValueError(
            "get_framework failed to find a valid library from the inputs: "
            "{} {}".format(args, kwargs)
        )
    if verbosity.level > 0:
        verbosity.cprint("Using framework from type: {}".format(f))
    return f


def set_framework(framework: str):
    """Sets `framework` to be the global framework.

    Examples
    --------
    If we set the global framework to be numpy, then subsequent calls to ivy functions
    will be called from Ivy's numpy backend:

    >>> ivy.set_framework("numpy")
    >>> native = ivy.native_array([1])
    >>> print(type(native))
    <class 'numpy.ndarray'>

    Or with jax as the global framework:

    >>> ivy.set_framework("jax")
    >>> native = ivy.native_array([1])
    >>> print(type(native))
    <class 'jaxlib.xla_extension.DeviceArray'>

    """
    global ivy_original_dict
    global ivy_original_fn_dict
    if not framework_stack:
        ivy_original_dict = ivy.__dict__.copy()
    if isinstance(framework, str):
        temp_stack = list()
        while framework_stack:
            temp_stack.append(unset_framework())
        framework = importlib.import_module(_framework_dict[framework])
        for fw in reversed(temp_stack):
            framework_stack.append(fw)
    if framework.current_framework_str() == "numpy":
        ivy.set_default_device("cpu")
    framework_stack.append(framework)
    ivy_original_fn_dict.clear()
    # loop through items in ivy dict and replace ivy's implementations `v` with the
    # appropriate backend implementation (backend specified by `framework`)
    for k, v in ivy_original_dict.items():
        if k not in framework.__dict__:
            if k in ivy.valid_dtypes:
                del ivy.__dict__[k]
                continue
            framework.__dict__[k] = v
        specific_v = framework.__dict__[k]
        if hasattr(v, "array_spec"):
            specific_v.array_spec = v.array_spec
        ivy.__dict__[k] = specific_v
        if isinstance(specific_v, collections.Hashable):
            try:
                ivy_original_fn_dict[specific_v] = v
            except TypeError:
                pass
    _wrap_functions()
    if verbosity.level > 0:
        verbosity.cprint("framework stack: {}".format(framework_stack))


def get_framework(framework: Optional[str] = None):
    """Returns Ivy's backend for `framework` if specified, or if it isn't specified it
    returns the Ivy backend associated with the current globally set framework.

    Parameters
    ----------
    framework
        The framework for which we want to retrieve Ivy's backend i.e. one of 'jax',
        'torch', 'tensorflow', 'numpy', 'mxnet'.

    Returns
    -------
    ret
        Ivy's backend for either `framework` or for the current global framework.

    Examples
    --------
    Global framework doesn't matter, if `framework` argument has been specified:

    >>> ivy.set_framework("jax")
    >>> ivy_np = ivy.get_framework("numpy")
    >>> print(ivy_np)
    <module 'ivy.functional.backends.numpy' from '/ivy/ivy/functional/backends/numpy/__init__.py'>   # noqa

    If framework isn't specified, the global framework is used:

    >>> ivy.set_framework("jax")
    >>> ivy_jax = ivy.get_framework()
    >>> print(ivy_jax)
    <module 'ivy.functional.backends.jax' from '/ivy/ivy/functional/backends/jax/__init__.py'>   # noqa

    """
    # ToDo: change this so that it doesn't depend at all on the global ivy. Currently
    #  all framework-agnostic implementations returned in this module will still
    #  use the global ivy backend.
    global ivy_original_dict
    if not framework_stack:
        ivy_original_dict = ivy.__dict__.copy()
    # current global framework is retrieved if framework isn't specified,
    # otherwise `framework` argument will be used
    if framework is None:
        framework = ivy.current_framework()
    elif isinstance(framework, str):
        framework = importlib.import_module(_framework_dict[framework])
    for k, v in ivy_original_dict.items():
        if k not in framework.__dict__:
            framework.__dict__[k] = v
    return framework


def unset_framework():
    """Unsets the current global framework, and adjusts the ivy dict such that either
    a previously set global framework is then used as the backend, otherwise we return
    to Ivy's implementations.

    Returns
    -------
    ret
        the framework that was unset, or None if there was no set global framework.

    Examples
    --------
    Torch is the last set framework hence is the backend framework used here:

    >>> ivy.set_framework("tensorflow")
    >>> ivy.set_framework("torch")
    >>> x = ivy.native_array([1])
    >>> print(type(x))
    <class 'torch.Tensor'>

    However if `unset_framework` is called before `ivy.native_array` then tensorflow
    will become the current framework and any torch backend implementations in the
    Ivy dict will be swapped with the tensorflow implementation:

    >>> ivy.set_framework("tensorflow")
    >>> ivy.set_framework("torch")
    >>> ivy.unset_framework()
    >>> x = ivy.native_array([1])
    >>> print(type(x))
    <class 'tensorflow.python.framework.ops.EagerTensor'>

    """
    framework = None
    # if the framework stack is empty, nothing is done and we just return `None`
    if framework_stack:
        _unwrap_functions()
        framework = framework_stack.pop(-1)  # remove last framework from the stack
        if framework.current_framework_str() == "numpy":
            ivy.unset_default_device()
        # the new framework is the framework that was set before the one we just removed
        # from the stack, or Ivy if there was no previously set framework
        new_framework_dict = (
            framework_stack[-1].__dict__ if framework_stack else ivy_original_dict
        )
        for k, v in new_framework_dict.items():
            ivy.__dict__[k] = v
    if verbosity.level > 0:
        verbosity.cprint("framework stack: {}".format(framework_stack))
    if framework_stack:
        _wrap_functions()
    return framework


def clear_framework_stack():
    while framework_stack:
        unset_framework()


# Framework Getters #
# ------------------#


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


def choose_random_framework(excluded=None):
    excluded = list() if excluded is None else excluded
    while True:
        if len(excluded) == 5:
            raise Exception(
                "Unable to select framework, all backends are either excluded "
                "or not installed."
            )
        f = np.random.choice(
            [f_srt for f_srt in list(FW_DICT.keys()) if f_srt not in excluded]
        )
        if f is None:
            excluded.append(f)
            continue
        else:
            print("\nselected framework: {}\n".format(f))
            return f
