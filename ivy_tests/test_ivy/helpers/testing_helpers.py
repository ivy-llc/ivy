# general
import importlib
import inspect
from typing import List

from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend
from .hypothesis_helpers import number_helpers as nh
from .globals import TestData
from . import test_parameter_flags as pf
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    BuiltInstanceStrategy,
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
    BuiltGradientStrategy,
    BuiltContainerStrategy,
    BuiltWithOutStrategy,
    BuiltInplaceStrategy,
    BuiltCompileStrategy,
)
from ivy_tests.test_ivy.helpers.structs import FrontendMethodData
from ivy_tests.test_ivy.helpers.available_frameworks import (
    available_frameworks,
    ground_truth,
)
from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import (
    _dtype_kind_keys,
    _get_type_dict,
)

ground_truth = ground_truth()


cmd_line_args = (
    "with_out",
    "instance_method",
    "test_gradients",
    "test_compile",
)
cmd_line_args_lists = (
    "as_variable",
    "native_array",
    "container",
)


@st.composite
def num_positional_args_method(draw, *, method):
    """
    Draws an integers randomly from the minimum and maximum number of positional
    arguments a given method can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    method
        callable method

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.
    """
    total, num_positional_only, num_keyword_only = (0, 0, 0)
    for param in inspect.signature(method).parameters.values():
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


@st.composite
def num_positional_args(draw, *, fn_name: str = None):
    """
    Draws an integers randomly from the minimum and maximum number of positional
    arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a
        given data-set (ex. list).
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


def _get_method_supported_devices_dtypes(
    method_name: str, class_module: str, class_name: str
):
    """
    Get supported devices and data types for a method in Ivy API
    Parameters
    ----------
    method_name
        Name of the method in the class

    class_module
        Name of the class module

    class_name
        Name of the class

    Returns
    -------
    Returns a dictonary containing supported device types and its supported data types
    for the method
    """
    supported_device_dtypes = {}
    backends = available_frameworks()
    for b in backends:  # ToDo can optimize this ?
        ivy.set_backend(b)
        _fn = getattr(class_module.__dict__[class_name], method_name)
        devices_and_dtypes = ivy.function_supported_devices_and_dtypes(_fn)
        organized_dtypes = {}
        for device in devices_and_dtypes.keys():
            organized_dtypes[device] = _partition_dtypes_into_kinds(
                ivy, devices_and_dtypes[device]
            )
        supported_device_dtypes[b] = organized_dtypes
        ivy.unset_backend()
    return supported_device_dtypes


def _get_supported_devices_dtypes(fn_name: str, fn_module: str):
    """
    Get supported devices and data types for a function in Ivy API
    Parameters
    ----------
    fn_name
        Name of the function

    fn_module
        Full import path of the function module

    Returns
    -------
    Returns a dictonary containing supported device types and its supported data types
    for the function
    """
    supported_device_dtypes = {}

    # This is for getting a private function from numpy frontend where we have
    # a ufunc object as we can't refer to them as functions
    if fn_module == "ivy.functional.frontends.numpy":
        fn_module_ = np_frontend
        if isinstance(getattr(fn_module_, fn_name), fn_module_.ufunc):
            fn_name = "_" + fn_name

    backends = available_frameworks()
    for b in backends:  # ToDo can optimize this ?
        ivy.set_backend(b)
        _tmp_mod = importlib.import_module(fn_module)
        _fn = _tmp_mod.__dict__[fn_name]
        devices_and_dtypes = ivy.function_supported_devices_and_dtypes(_fn)
        organized_dtypes = {}
        for device in devices_and_dtypes.keys():
            organized_dtypes[device] = _partition_dtypes_into_kinds(
                ivy, devices_and_dtypes[device]
            )
        supported_device_dtypes[b] = organized_dtypes
        ivy.unset_backend()
    return supported_device_dtypes


def _partition_dtypes_into_kinds(framework, dtypes):
    partitioned_dtypes = {}
    for kind in _dtype_kind_keys:
        partitioned_dtypes[kind] = set(_get_type_dict(framework, kind)).intersection(
            dtypes
        )
    return partitioned_dtypes


# Decorators


def handle_test(
    *,
    fn_tree: str = None,
    ground_truth_backend: str = ground_truth,
    number_positional_args=None,
    test_instance_method=BuiltInstanceStrategy,
    test_with_out=BuiltWithOutStrategy,
    test_gradients=BuiltGradientStrategy,
    test_compile=BuiltCompileStrategy,
    as_variable_flags=BuiltAsVariableStrategy,
    native_array_flags=BuiltNativeArrayStrategy,
    container_flags=BuiltContainerStrategy,
    **_given_kwargs,
):
    """
    A test wrapper for Ivy functions.
    Sets the required test globals and creates test flags strategies.

    Parameters
    ----------
    fn_tree
        Full function import path

    ground_truth_backend
        The framework to assert test results are equal to

    number_positional_args
        A search strategy for determining the number of positional arguments to be
        passed to the function

    test_instance_method
        A search strategy that generates a boolean to test instance methods

    test_with_out
        A search strategy that generates a boolean to test the function with an `out`
        parameter

    test_gradients
        A search strategy that generates a boolean to test the function with arrays as
        gradients

    test_compile
        A search strategy that generates a boolean to graph compile and test the
        function

    as_variable_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a Variable array

    native_array_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a native array

    container_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a Container
    """
    is_fn_tree_provided = fn_tree is not None
    if is_fn_tree_provided:
        fn_tree = "ivy." + fn_tree
    is_hypothesis_test = len(_given_kwargs) != 0

    possible_arguments = {"ground_truth_backend": st.just(ground_truth_backend)}
    if is_hypothesis_test and is_fn_tree_provided:
        # Use the default strategy
        if number_positional_args is None:
            number_positional_args = num_positional_args(fn_name=fn_tree)
        # Generate the test flags strategy
        possible_arguments["test_flags"] = pf.function_flags(
            num_positional_args=number_positional_args,
            instance_method=test_instance_method,
            with_out=test_with_out,
            test_gradients=test_gradients,
            test_compile=test_compile,
            as_variable=as_variable_flags,
            native_arrays=native_array_flags,
            container_flags=container_flags,
        )

    def test_wrapper(test_fn):
        if is_fn_tree_provided:
            callable_fn, fn_name, fn_mod = _import_fn(fn_tree)
            supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)
            possible_arguments["fn_name"] = st.just(fn_name)

        # If a test is not a Hypothesis test, we only set the test global data
        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            # Check if these arguments are being asked for
            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                _given_kwargs[key] = possible_arguments[key]
            # Wrap the test with the @given decorator
            wrapped_test = given(**_given_kwargs)(test_fn)
        else:
            wrapped_test = test_fn

        # Set the test data to be used by test helpers
        if is_fn_tree_provided:
            wrapped_test.test_data = TestData(
                test_fn=wrapped_test,
                fn_tree=fn_tree,
                fn_name=fn_name,
                supported_device_dtypes=supported_device_dtypes,
            )
        wrapped_test.ground_truth_backend = ground_truth_backend
        wrapped_test._ivy_test = True

        return wrapped_test

    return test_wrapper


def handle_frontend_test(
    *,
    fn_tree: str,
    aliases: List[str] = None,
    number_positional_args=None,
    test_with_out=BuiltWithOutStrategy,
    test_inplace=BuiltInplaceStrategy,
    as_variable_flags=BuiltAsVariableStrategy,
    native_array_flags=BuiltNativeArrayStrategy,
    **_given_kwargs,
):
    """
    A test wrapper for Ivy frontend functions.
    Sets the required test globals and creates test flags strategies.

    Parameters
    ----------
    fn_tree
        Full function import path

    number_positional_args
        A search strategy for determining the number of positional arguments to be
        passed to the function

    test_inplace
        A search strategy that generates a boolean to test the method with `inplace`
        update

    test_with_out
        A search strategy that generates a boolean to test the function with an `out`
        parameter

    as_variable_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a Variable array

    native_array_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a native array
    """
    fn_tree = "ivy.functional.frontends." + fn_tree
    if aliases is not None:
        for i in range(len(aliases)):
            aliases[i] = "ivy.functional.frontends." + aliases[i]
    is_hypothesis_test = len(_given_kwargs) != 0

    if is_hypothesis_test:
        # Use the default strategy
        if number_positional_args is None:
            number_positional_args = num_positional_args(fn_name=fn_tree)
        # Generate the test flags strategy
        test_flags = pf.frontend_function_flags(
            num_positional_args=number_positional_args,
            with_out=test_with_out,
            inplace=test_inplace,
            as_variable=as_variable_flags,
            native_arrays=native_array_flags,
        )

    def test_wrapper(test_fn):
        callable_fn, fn_name, fn_mod = _import_fn(fn_tree)
        supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)

        # If a test is not a Hypothesis test, we only set the test global data
        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            # Check if these arguments are being asked for
            possible_arguments = {
                "test_flags": test_flags,
                "fn_tree": st.sampled_from([fn_tree] + aliases)
                if aliases is not None
                else st.just(fn_tree),
            }
            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                # extend Hypothesis given kwargs with our stratigies
                _given_kwargs[key] = possible_arguments[key]
            # Wrap the test with the @given decorator
            wrapped_test = given(**_given_kwargs)(test_fn)
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


def handle_method(
    *,
    method_tree: str = None,
    ground_truth_backend: str = ground_truth,
    test_gradients=BuiltGradientStrategy,
    test_compile=BuiltCompileStrategy,
    init_num_positional_args=None,
    init_native_arrays=BuiltNativeArrayStrategy,
    init_as_variable_flags=BuiltAsVariableStrategy,
    init_container_flags=BuiltContainerStrategy,
    method_num_positional_args=None,
    method_native_arrays=BuiltNativeArrayStrategy,
    method_as_variable_flags=BuiltAsVariableStrategy,
    method_container_flags=BuiltContainerStrategy,
    **_given_kwargs,
):
    """
    A test wrapper for Ivy methods.
    Sets the required test globals and creates test flags strategies.

    Parameters
    ----------
    method_tree
        Full method import path

    ground_truth_backend
        The framework to assert test results are equal to
    """
    is_method_tree_provided = method_tree is not None
    if is_method_tree_provided:
        method_tree = "ivy." + method_tree
    is_hypothesis_test = len(_given_kwargs) != 0
    possible_arguments = {
        "ground_truth_backend": st.just(ground_truth_backend),
        "test_gradients": test_gradients,
        "test_compile": test_compile,
    }

    if is_hypothesis_test and is_method_tree_provided:
        callable_method, method_name, _, class_name, method_mod = _import_method(
            method_tree
        )

        if init_num_positional_args is None:
            init_num_positional_args = num_positional_args(
                fn_name=class_name + ".__init__"
            )

        possible_arguments["init_flags"] = pf.method_flags(
            num_positional_args=init_num_positional_args,
            as_variable=init_as_variable_flags,
            native_arrays=init_native_arrays,
            container_flags=init_container_flags,
        )

        if method_num_positional_args is None:
            method_num_positional_args = num_positional_args_method(
                method=callable_method
            )

        possible_arguments["method_flags"] = pf.method_flags(
            num_positional_args=method_num_positional_args,
            as_variable=method_as_variable_flags,
            native_arrays=method_native_arrays,
            container_flags=method_container_flags,
        )

    def test_wrapper(test_fn):
        if is_method_tree_provided:
            supported_device_dtypes = _get_method_supported_devices_dtypes(
                method_name, method_mod, class_name
            )
            possible_arguments["class_name"] = st.just(class_name)
            possible_arguments["method_name"] = st.just(method_name)

        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            filtered_args = set(param_names).intersection(possible_arguments.keys())

            for key in filtered_args:
                # extend Hypothesis given kwargs with our strategies
                _given_kwargs[key] = possible_arguments[key]

            wrapped_test = given(**_given_kwargs)(test_fn)
        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=method_tree,
            fn_name=method_name,
            supported_device_dtypes=supported_device_dtypes,
            is_method=True,
        )

        wrapped_test.ground_truth_backend = ground_truth_backend
        wrapped_test._ivy_test = True

        return wrapped_test

    return test_wrapper


def handle_frontend_method(
    *,
    class_tree: str,
    init_tree: str,
    method_name: str,
    init_num_positional_args=None,
    init_native_arrays=BuiltNativeArrayStrategy,
    init_as_variable_flags=BuiltAsVariableStrategy,
    method_num_positional_args=None,
    method_native_arrays=BuiltNativeArrayStrategy,
    method_as_variable_flags=BuiltAsVariableStrategy,
    **_given_kwargs,
):
    """
    A test wrapper for Ivy frontends methods.
    Sets the required test globals and creates test flags strategies.

    Parameters
    ----------
    class_tree
        Full class import path

    init_tree
        Full import path for the function used to create the class

    method_name
        Name of the method
    """
    split_index = init_tree.rfind(".")
    framework_init_module = init_tree[:split_index]
    ivy_init_module = f"ivy.functional.frontends.{init_tree[:split_index]}"
    init_name = init_tree[split_index + 1 :]
    init_tree = f"ivy.functional.frontends.{init_tree}"
    is_hypothesis_test = len(_given_kwargs) != 0

    split_index = class_tree.rfind(".")
    class_module_path, class_name = (
        class_tree[:split_index],
        class_tree[split_index + 1 :],
    )
    class_module = importlib.import_module(class_module_path)
    method_class = getattr(class_module, class_name)

    if is_hypothesis_test:
        callable_method = getattr(method_class, method_name)
        if init_num_positional_args is None:
            init_num_positional_args = num_positional_args(fn_name=init_tree[4:])

        if method_num_positional_args is None:
            method_num_positional_args = num_positional_args_method(
                method=callable_method
            )

    def test_wrapper(test_fn):
        supported_device_dtypes = _get_method_supported_devices_dtypes(
            method_name, class_module, class_name
        )

        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            init_flags = pf.frontend_method_flags(
                num_positional_args=init_num_positional_args,
                as_variable=init_as_variable_flags,
                native_arrays=init_native_arrays,
            )

            method_flags = pf.frontend_method_flags(
                num_positional_args=method_num_positional_args,
                as_variable=method_as_variable_flags,
                native_arrays=method_native_arrays,
            )
            try:
                ivy_init_modules = importlib.import_module(ivy_init_module)
            except Exception:
                ivy_init_modules = str(ivy_init_module)
            try:
                framework_init_modules = importlib.import_module(framework_init_module)
            except Exception:
                framework_init_modules = str(framework_init_module)
            frontend_helper_data = FrontendMethodData(
                ivy_init_module=ivy_init_modules,
                framework_init_module=framework_init_modules,
                init_name=init_name,
                method_name=method_name,
            )

            possible_arguments = {
                "init_flags": init_flags,
                "method_flags": method_flags,
                "frontend_method_data": st.just(frontend_helper_data),
            }

            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                # extend Hypothesis given kwargs with our strategies
                _given_kwargs[key] = possible_arguments[key]
            wrapped_test = given(**_given_kwargs)(test_fn)
        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=f"{init_tree}.{method_name}",
            fn_name=method_name,
            supported_device_dtypes=supported_device_dtypes,
            is_method=[method_name, class_tree, split_index],
        )

        return wrapped_test

    return test_wrapper


@st.composite
def seed(draw):
    return draw(st.integers(min_value=0, max_value=2**8 - 1))
