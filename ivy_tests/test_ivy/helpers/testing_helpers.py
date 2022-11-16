# general
import importlib
import inspect
import typing
from functools import partial

from hypothesis import given, strategies as st

# local
import ivy
from ivy_tests.test_ivy import conftest as cfg  # TODO temporary
from .hypothesis_helpers import number_helpers as nh
from .globals import TestData
from . import test_parameter_flags as pf

cmd_line_args = (
    "with_out",
    "instance_method",
    "test_gradients",
)
cmd_line_args_lists = (
    "as_variable",
    "native_array",
    "container",
)


@st.composite
def num_positional_args(draw, *, fn_name: str = None):
    """Draws an integers randomly from the minimum and maximum number of positional
    arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    fn_name
        name of the function.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.

    Examples
    --------
    @given(
        num_positional_args=num_positional_args(fn_name="floor_divide")
    )
    @given(
        num_positional_args=num_positional_args(fn_name="add")
    )
    """
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    fn = None
    for i, fn_name_key in enumerate(fn_name.split(".")):
        if i == 0:
            fn = ivy.__dict__[fn_name_key]
        else:
            fn = fn.__dict__[fn_name_key]
    for param in inspect.signature(fn).parameters.values():
        if param.name == "self":
            continue
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
            num_keyword_only += 1
        elif param.kind == param.VAR_KEYWORD:
            num_keyword_only += 1
    return draw(
        nh.ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


# Decorators helpers


def _import_fn(fn_tree: str):
    """
    Imports a function from function tree string
    Parameters
    ----------
    fn_tree
        Full function tree without "ivy" root
        example: "functional.backends.jax.creation.arange".

    Returns
    -------
    Returns fn_name, imported module, callable function
    """
    split_index = fn_tree.rfind(".")
    fn_name = fn_tree[split_index + 1 :]
    module_to_import = fn_tree[:split_index]
    mod = importlib.import_module(module_to_import)
    callable_fn = mod.__dict__[fn_name]
    return callable_fn, fn_name, module_to_import


def _generate_shared_test_flags(
    param_names: list, _given_kwargs: dict, fn_tree: str, fn: callable
):
    """
    Generates flags that all tests use.

    Returns
    -------
    shared flags that all tests use.
    """
    if "num_positional_args" in param_names:
        _given_kwargs["num_positional_args"] = num_positional_args(fn_name=fn_tree)
    for flag_key, flag_value in cfg.GENERAL_CONFIG_DICT.items():
        if flag_key in param_names:
            _given_kwargs[flag_key] = st.just(flag_value)
    for flag in cfg.UNSET_TEST_CONFIG["list"]:
        if flag in param_names:
            _given_kwargs[flag] = st.lists(st.booleans(), min_size=1, max_size=1)
    for flag in cfg.UNSET_TEST_CONFIG["flag"]:
        if flag in param_names:
            _given_kwargs[flag] = st.booleans()
    # Override with_out to be compatible
    if "with_out" in param_names:
        for k in inspect.signature(fn).parameters.keys():
            if k.endswith("out"):
                break
        else:
            _given_kwargs["with_out"] = st.just(False)
    return _given_kwargs


def _get_method_supported_devices_dtypes(fn_name: str, fn_module: str, class_name: str):
    supported_device_dtypes = {}
    backends = ["numpy", "jax", "tensorflow", "torch"]  # TODO temporary
    for b in backends:  # ToDo can optimize this ?
        ivy.set_backend(b)
        _tmp_mod = importlib.import_module(fn_module)
        _fn = getattr(_tmp_mod.__dict__[class_name], fn_name)
        supported_device_dtypes[b] = ivy.function_supported_devices_and_dtypes(_fn)
        ivy.unset_backend()
    return supported_device_dtypes


def _get_supported_devices_dtypes(fn_name: str, fn_module: str):
    supported_device_dtypes = {}
    backends = ["numpy", "jax", "tensorflow", "torch"]  # TODO temporary
    for b in backends:  # ToDo can optimize this ?
        ivy.set_backend(b)
        _tmp_mod = importlib.import_module(fn_module)
        _fn = _tmp_mod.__dict__[fn_name]
        supported_device_dtypes[b] = ivy.function_supported_devices_and_dtypes(_fn)
        ivy.unset_backend()
    return supported_device_dtypes


# Decorators

possible_fixtures = ["backend_fw", "on_device"]


def handle_test(*, fn_tree: str, **_given_kwargs):
    fn_tree = "ivy." + fn_tree
    is_hypothesis_test = len(_given_kwargs) != 0
    given_kwargs = _given_kwargs

    def test_wrapper(test_fn):
        callable_fn, fn_name, fn_mod = _import_fn(fn_tree)
        param_names = inspect.signature(test_fn).parameters.keys()
        supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)

        # No Hypothesis @given is used
        if is_hypothesis_test:
            _given_kwargs = _generate_shared_test_flags(
                param_names, given_kwargs, fn_tree, callable_fn
            )
            for flag in cfg.UNSET_TEST_API_CONFIG["list"]:
                if flag in param_names:
                    _given_kwargs[flag] = st.lists(
                        st.booleans(), min_size=1, max_size=1
                    )
            for flag in cfg.UNSET_TEST_API_CONFIG["flag"]:
                if flag in param_names:
                    _given_kwargs[flag] = st.booleans()

            wrapped_test = given(**_given_kwargs)(test_fn)
            if "fn_name" in param_names:
                _name = wrapped_test.__name__
                wrapped_test = partial(wrapped_test, fn_name=fn_name)
                wrapped_test.__name__ = _name
        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=fn_tree,
            fn_name=fn_name,
            supported_device_dtypes=supported_device_dtypes,
        )

        return wrapped_test

    return test_wrapper


possible_fixtures_frontends = ["on_device", "frontend"]


def handle_frontend_test(*, fn_tree: str, **_given_kwargs):
    fn_tree = "ivy.functional.frontends." + fn_tree
    is_hypothesis_test = len(_given_kwargs) != 0
    given_kwargs = _given_kwargs

    def test_wrapper(test_fn):
        callable_fn, fn_name, fn_mod = _import_fn(fn_tree)
        supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)

        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            _given_kwargs = _generate_shared_test_flags(
                param_names, given_kwargs, fn_tree, callable_fn
            )
            wrapped_test = given(**_given_kwargs)(test_fn)
            if "fn_tree" in param_names:
                _name = wrapped_test.__name__
                wrapped_test = partial(wrapped_test, fn_tree=fn_tree)
                wrapped_test.__name__ = _name
        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=fn_tree,
            fn_name=fn_name,
            supported_device_dtypes=supported_device_dtypes,
        )

        return wrapped_test

    return test_wrapper


def _import_method(method_tree: str):
    split_index = method_tree.rfind(".")
    class_tree, method_name = method_tree[:split_index], method_tree[split_index + 1 :]
    split_index = class_tree.rfind(".")
    mod_to_import, class_name = class_tree[:split_index], class_tree[split_index + 1 :]
    _mod = importlib.import_module(mod_to_import)
    _class = _mod.__getattribute__(class_name)
    _method = getattr(_class, method_name)
    return _method, method_name, _class, class_name, _mod


def handle_method(*, method_tree, **_given_kwargs):
    method_tree = "ivy." + method_tree
    is_hypothesis_test = len(_given_kwargs) != 0

    def test_wrapper(test_fn):
        callable_method, method_name, class_, class_name, method_mod = _import_method(
            method_tree
        )
        supported_device_dtypes = _get_method_supported_devices_dtypes(
            method_name, method_mod.__name__, class_name
        )

        if is_hypothesis_test:
            fn_args = typing.get_type_hints(test_fn)

            for k, v in fn_args.items():
                if (
                    v is pf.NativeArrayFlags
                    or v is pf.ContainerFlags
                    or v is pf.AsVariableFlags
                ):
                    _given_kwargs[k] = st.lists(st.booleans(), min_size=1, max_size=1)
                elif v is pf.NumPositionalArg:
                    if k.startswith("method"):
                        _given_kwargs[k] = num_positional_args(
                            fn_name=f"{class_name}.{method_name}"
                        )
                    else:
                        _given_kwargs[k] = num_positional_args(
                            fn_name=class_name + ".__init__"
                        )

            wrapped_test = given(**_given_kwargs)(test_fn)
            _name = wrapped_test.__name__
            wrapped_test = partial(
                wrapped_test, class_name=class_name, method_name=method_name
            )
            wrapped_test.__name__ = _name
        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=method_tree,
            fn_name=method_name,
            supported_device_dtypes=supported_device_dtypes,
        )

        return wrapped_test

    return test_wrapper


def handle_frontend_method(*, method_tree, **_given_kwargs):
    method_tree = "ivy.functional.frontends." + method_tree
    is_hypothesis_test = len(_given_kwargs) != 0

    def test_wrapper(test_fn):
        callable_method, method_name, class_, class_name, method_mod = _import_method(
            method_tree
        )
        supported_device_dtypes = _get_method_supported_devices_dtypes(
            method_name, callable_method.__module__, class_name
        )

        if is_hypothesis_test:
            fn_args = typing.get_type_hints(test_fn)

            for k, v in fn_args.items():
                if (
                    v is pf.NativeArrayFlags
                    or v is pf.ContainerFlags
                    or v is pf.AsVariableFlags
                ):
                    _given_kwargs[k] = st.lists(st.booleans(), min_size=1, max_size=1)
                elif v is pf.NumPositionalArg:
                    if k.startswith("method"):
                        _given_kwargs[k] = num_positional_args(
                            f"{class_name}.{method_name}"
                        )
                    else:
                        _given_kwargs[k] = num_positional_args(class_name + ".__init__")

            wrapped_test = given(**_given_kwargs)(test_fn)
            _name = wrapped_test.__name__
            wrapped_test = partial(wrapped_test, class_=class_, method_name=method_name)
            wrapped_test.__name__ = _name
        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=method_tree,
            fn_name=method_name,
            supported_device_dtypes=supported_device_dtypes,
        )

        return wrapped_test

    return test_wrapper


@st.composite
def seed(draw):
    return draw(st.integers(min_value=0, max_value=2**8 - 1))
