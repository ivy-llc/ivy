import importlib
import logging
import os
import re
from types import FunctionType, ModuleType

import ivy
from ivy.func_wrapper import _wrap_function
from ivy.utils.exceptions import IvyException

_backends_subpackage_path = "ivy.functional.backends"
_sub_backend_dict: dict[str, str] = {}
_backend_to_sub_backends_dict: dict[str, list] = {}


def _detect_sub_backends_dynamically():
    for backend in os.listdir(
        os.path.join(
            ivy.__path__[0].rpartition(os.path.sep)[0],  # type: ignore
            _backends_subpackage_path.replace(".", os.path.sep),
        )
    ):
        if not backend[0].isalpha():
            continue

        sub_backends_dir = os.path.join(
            ivy.__path__[0].rpartition(os.path.sep)[0],
            _backends_subpackage_path.replace(".", os.path.sep),
            backend,
            "sub_backends",
        )
        for sub_backend in os.listdir(sub_backends_dir):
            if not sub_backend[0].isalpha():
                continue
            _sub_backend_dict[sub_backend] = (
                f"{_backends_subpackage_path}.{backend}.sub_backends.{sub_backend}"
            )
            try:
                _backend_to_sub_backends_dict[backend].append(sub_backend)
            except KeyError:
                _backend_to_sub_backends_dict[backend] = [sub_backend]


_detect_sub_backends_dynamically()


def _get_all_sub_backends():
    sub_backends = []
    for v in _backend_to_sub_backends_dict.values():
        sub_backends.extend(v)
    return sub_backends


_all_sub_backends = _get_all_sub_backends()


original_backend_dict = None


# version specific sub-backend setting
def set_sub_backend_to_specific_version(sub_backend):
    f = str(sub_backend.__name__)
    f_sub = f[f.index("sub_backends") + 13 :]
    f_back = f[f.index("backends") + 9 : f.index(".sub_backends")]
    f_sub = importlib.import_module(f_sub)
    f_back = importlib.import_module(f_back)
    f_sub_version = f_sub.__version__
    f_back_version = f_back.__version__

    for key in list(sub_backend.__dict__):
        if "_v_" in key:
            orig_name = fn_name_from_version_specific_fn_name_sub_backend(
                key, f_sub_version, f_back_version
            )
            if orig_name:
                sub_backend.__dict__[orig_name] = sub_backend.__dict__[key]
                sub_backend.__dict__[orig_name].__name__ = orig_name


def fn_name_from_version_specific_fn_name(name, version):
    """
    Parameters
    ----------
    name
        the version specific name of the function for which the version support
        is to be provided.
    version
        the version of the current framework for which the support is to be
        provided, the version is inferred by importing the framework
    Returns
    -------
        the name of the original function which will then point to the version
        specific function

    """
    # TODO: add tests
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
        if version >= version_start:
            return name[0:i]
    else:
        i = name.index("_v_")
        e = name.index("_and_")
        version_start = name[i + 3 : e]
        version_start = tuple(map(int, version_start.split("p")))
        if version <= version_start:
            return name[0:i]


def fn_name_from_version_specific_fn_name_sub_backend(
    name, sub_backend_version, backend_version
):
    """
    Parameters
    ----------
    name
        the version specific name of the function for which the version support
        is to be provided.
    version
        the version of the current framework for which the support is to be
        provided, the version is inferred by importing the framework
    Returns
    -------
        the name of the original function which will then point to the version
        specific function

    """
    # TODO: add tests
    sub_version = str(sub_backend_version)
    back_version = str(backend_version)
    if sub_version.find("+") != -1:
        sub_version = tuple(map(int, sub_version[: sub_version.index("+")].split(".")))
    else:
        sub_version = tuple(map(int, sub_version.split(".")))

    if back_version.find("+") != -1:
        back_version = tuple(
            map(int, back_version[: back_version.index("+")].split("."))
        )
    else:
        back_version = tuple(map(int, back_version.split(".")))
    v_occurences = [m.start() for m in re.finditer("_v_", name)]
    fn_name_1 = name[: v_occurences[1] + 3]
    fn_name_2 = name[: v_occurences[0]] + name[v_occurences[1] :]
    ret_1 = fn_name_from_version_specific_fn_name(fn_name_1, sub_backend_version)
    ret_2 = fn_name_from_version_specific_fn_name(fn_name_2, backend_version)
    if ret_1 == ret_2:
        return name[: v_occurences[0]]


def set_sub_backend(sub_backend_str: str):
    if ivy.backend == "":
        logging.warning("You must set a backend first")
        return

    if ivy.current_backend_str() not in _backend_to_sub_backends_dict.keys():
        logging.warning(
            f"backend {ivy.current_backend_str()} does not have any"
            " supported sub_backends"
        )
        return

    if sub_backend_str not in _all_sub_backends:
        raise IvyException(
            "sub_backend must be one from"
            f" {_backend_to_sub_backends_dict[ivy.current_backend_str()]}"
        )

    if sub_backend_str not in _backend_to_sub_backends_dict[ivy.current_backend_str()]:
        logging.warning(
            f"{ivy.current_backend_str()} does not support"
            f" {sub_backend_str} as a sub_backend"
        )
        return

    if sub_backend_str in ivy.current_sub_backends():
        return

    global original_backend_dict
    if original_backend_dict is None:
        original_backend_dict = ivy.__dict__.copy()
    sub_backend = ivy.utils.dynamic_import.import_module(
        _sub_backend_dict[sub_backend_str]
    )
    set_sub_backend_to_specific_version(sub_backend)
    _set_sub_backend_as_ivy(ivy.__dict__.copy(), ivy, sub_backend)
    ivy.current_backend().sub_backends._current_sub_backends.append(sub_backend_str)


# this is very similar to _set_backend_as_ivy in handler.py, with a minor change
def _set_sub_backend_as_ivy(
    original: dict, target: ModuleType, sub_backend: ModuleType
):
    backend_str = ivy.current_backend_str()
    for k, v in original.items():
        if k not in sub_backend.__dict__ and not k.startswith("__"):
            target.__dict__[k] = v
        if (
            k in sub_backend.__dict__
            and not k.startswith("__")
            and isinstance(v, FunctionType)
        ):
            target.__dict__[k] = _wrap_function(
                key=k, to_wrap=sub_backend.__dict__[k], original=v, compositional=False
            )
        elif (
            k in sub_backend.__dict__
            and not k.startswith("__")
            and isinstance(v, ModuleType)
        ):
            # we are creating a module to avoid inplace updating
            # the sub_backends dict's modules, this happens when
            # unsetting the sub_backend as we partially update the modules
            mod = ModuleType(k)
            mod.__name__ = v.__name__
            mod.__file__ = v.__file__
            target.__dict__[k] = mod
        if (
            isinstance(v, ModuleType)
            and "ivy.functional." in v.__name__
            and os.path.join("{}", "__init__.py").format(backend_str) not in v.__file__
            and k in sub_backend.__dict__
        ):
            _set_sub_backend_as_ivy(
                v.__dict__,
                target.__dict__[k],
                sub_backend.__dict__[k],
            )


def unset_sub_backend(sub_backend_str: str):
    if sub_backend_str not in ivy.current_sub_backends():
        return
    global original_backend_dict

    # The sub-backend is cached so this is fast
    sub_backend = ivy.utils.dynamic_import.import_module(
        _sub_backend_dict[sub_backend_str]
    )
    _unset_sub_backend_from_ivy(
        original_backend_dict, ivy, sub_backend, sub_backend.name
    )
    ivy.current_backend().sub_backends._current_sub_backends.remove(sub_backend_str)


def _unset_sub_backend_from_ivy(
    original: dict, target: ModuleType, sub_backend: ModuleType, sub_backend_str: str
):
    backend_str = ivy.current_backend_str()
    for k, v in sub_backend.__dict__.items():
        if k in target.__dict__:
            if (
                isinstance(v, FunctionType)
                and sub_backend_str in f"sub_backends.{sub_backend_str}" in v.__module__
            ):
                target.__dict__[k] = original[k]
            if (
                isinstance(v, ModuleType)
                and "ivy.functional." in v.__name__
                and os.path.join("{}", "__init__.py").format(backend_str)
                not in v.__file__
            ):
                _unset_sub_backend_from_ivy(
                    original[k].__dict__,
                    target.__dict__[k],
                    sub_backend.__dict__[k],
                    sub_backend_str,
                )


def clear_sub_backends():
    if ivy.current_sub_backends():
        ivy.__dict__.update(original_backend_dict)
        ivy.current_backend().sub_backends._current_sub_backends = []


# This is only used in set_backend in handler.py
def _clear_current_sub_backends():
    global original_backend_dict
    original_backend_dict = None
    if ivy.current_sub_backends():
        ivy.current_backend().sub_backends._current_sub_backends = []


# this is overwritten when setting a backend
def available_sub_backends():
    for k, v in _backend_to_sub_backends_dict.items():
        print(f"backend: {k} supports sub_backends: {v}")


def find_available_sub_backends(sub_backends_loc):
    available_sub_backends = []
    for sub_backend in os.listdir(sub_backends_loc):
        if sub_backend.startswith("__") or not os.path.isdir(
            os.path.join(sub_backends_loc, sub_backend)
        ):
            continue

        elif importlib.util.find_spec(sub_backend):
            available_sub_backends.append(sub_backend)

    return available_sub_backends


def available_sub_backend_implementations(fn_name: str) -> list:
    """
    Return whether a sub-backend implementation is available for `fn_name`.

    Parameters
    ----------
    fn_name : str
        the object for which to check if a sub-backend implementation is available.

    Returns
    -------
    ret : list
        a list of sub-backend implementations available for `fn_name`.

    Examples
    --------
    >>> import ivy
    >>> ivy.set_backend('torch')
    >>> ivy.available_sub_backend_implementations("scaled_dot_product_attention")
    ['xformers']
    >>> ivy.set_backend('numpy')
    >>> ivy.available_sub_backend_implementations("scaled_dot_product_attention")
    []
    """
    sub_backends = ivy.current_backend().available_sub_backends()
    implementations = []
    for sub in sub_backends:
        try:
            sub_backend = ivy.utils.dynamic_import.import_module(_sub_backend_dict[sub])
        except ModuleNotFoundError:
            continue
        if fn_name in sub_backend.__dict__:
            implementations.append(sub)
    return implementations
